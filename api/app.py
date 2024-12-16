import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from substrateinterface import SubstrateInterface
import pandas as pd
from decimal import Decimal, getcontext, InvalidOperation
from tqdm import tqdm
import logging
from substrateinterface.exceptions import SubstrateRequestException
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current file
dir_path = Path(__file__).resolve().parent
# Go up one level to the project root
project_root = dir_path.parent

app = Flask(__name__,
            static_folder=os.path.join(project_root, 'static'),
            template_folder=os.path.join(project_root, 'templates'))

pd.options.display.max_rows = None
pd.options.display.max_columns = None

def formatBalance(amount):
    getcontext().prec = 30
    try:
        if amount is None:
            raise InvalidOperation
        decimalOutput = Decimal(amount) / Decimal(10**18)
    except InvalidOperation:
        print(f"Error converting amount: {amount}")
        return Decimal(0)
    return decimalOutput if amount != 0 else 0

def getCurrentEra(substrate):
    result = substrate.query("Staking", "CurrentEra")
    return int(str(result))

def validatorRewards(substrate, era):
    result = substrate.query("Staking", "ErasValidatorReward", [str(era)])
    return result

def validatorPref(substrate, era):
    result = substrate.query_map("Staking", "ErasValidatorPrefs", [str(era)])
    data = []
    for item in result:
        validator = str(item[0])
        commission = float(str(item[1]['commission'])) / 10**9 if float(str(item[1]['commission'])) >= 10 else 0.0
        data.append([validator, commission])
    return pd.DataFrame(data, columns=["Validator", "CommissionRate"])

def createInstance():
    substrate = SubstrateInterface(
        url="wss://mainnet-rpc.avail.so/ws",
        ss58_format=42,
        type_registry_preset='substrate-node-template'
    )
    return substrate

def createDf(substrate, era):
    result = substrate.query("Staking", "ErasRewardPoints", [str(era)])
    df = pd.DataFrame(result['individual'], columns=["Validator", "EraPoints"])
    df['Era'] = era
    df['EraPoints'] = df['EraPoints'].astype(str).astype(int)
    df['Validator'] = df['Validator'].astype(str)
    return df

def getStake(substrate, era):
    result = substrate.query_map("Staking", "ErasStakersOverview", [str(era)])
    data = []
    for item in result:
        validator, attrs = str(item[0]), item[1]
        total, own = formatBalance(str(attrs['total'])), formatBalance(str(attrs['own']))
        nominated = total - own
        data.append([validator, total, own, nominated, attrs['nominator_count']])
    return pd.DataFrame(data, columns=['Validator', 'TotalStake', 'OwnStake', 'NominatedStake', 'NomCount'])

def get_validator_identities(substrate, validators):
    identities = {}
    
    logger.info(f"Number of validators to fetch identities for: {len(validators)}")

    try:
        result = substrate.query_map('Identity', 'IdentityOf')

        for account, identity_info in result:
            if str(account.value) in validators:
                try:
                    info = identity_info.value[0]['info']
                    if 'display' in info and 'Raw' in info['display']:
                        validator_name = info['display']['Raw']
                        identities[str(account.value)] = validator_name
                        logger.info(f"Found identity for validator {account.value}: {validator_name}")
                    else:
                        identities[str(account.value)] = ''
                        logger.info(f"Validator Name not found in expected format for {account.value}")
                except Exception as e:
                    logger.warning(f"Error extracting validator name for {account.value}: {str(e)}")
                    identities[str(account.value)] = ''

        for validator in validators:
            if validator not in identities:
                identities[validator] = ''
                logger.info(f"No identity found for validator {validator}")

    except SubstrateRequestException as e:
        logger.error(f"A substrate request error occurred: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

    logger.info(f"Total identities processed: {len(identities)}")
    logger.info(f"Sample of identities: {dict(list(identities.items())[:5])}")
    return identities

def get_validator_history(substrate, validator, current_era):
    history = []
    start_era = max(0, current_era - 20)  # Changed to get 21 eras (current + past 20)
    
    # First, collect all eras where the validator was active
    active_eras = []
    for era in range(start_era, current_era + 1):
        try:
            result = substrate.query("Staking", "ErasRewardPoints", [str(era)])
            if result:
                individual_points = result['individual']
                for v, points in individual_points:
                    if str(v) == validator and int(str(points)) > 0:
                        blocks = int(str(points)) // 20
                        active_eras.append({
                            'era': era,
                            'blocks': blocks
                        })
                        break
        except Exception as e:
            logger.warning(f"Error fetching history for era {era}: {str(e)}")
            continue
    
    # If we have active eras, only return the last 21 active eras
    if active_eras:
        active_eras = active_eras[-21:] if len(active_eras) > 21 else active_eras
        return {
            'eras': [h['era'] for h in active_eras],
            'blocks': [h['blocks'] for h in active_eras]
        }
    
    return {
        'eras': [],
        'blocks': []
    }

def processData(substrate, era):
    try:
        df = createDf(substrate, era)
        df = pd.merge(df, getStake(substrate, era), on="Validator", how="inner")
        df = pd.merge(df, validatorPref(substrate, era), on="Validator", how="inner")

        identities = get_validator_identities(substrate, df['Validator'].tolist())
        df['ValidatorName'] = df['Validator'].map(identities)
        df['ValidatorStash'] = df['Validator'].astype(str)

        # Calculate BlocksProduced from EraPoints
        df['BlocksProduced'] = (df['EraPoints'] / 20).astype(int)

        totalRewardsRaw = validatorRewards(substrate, era)
        totalRewards = formatBalance(str(totalRewardsRaw)) if totalRewardsRaw is not None else Decimal(0)

        totalEraPoints = df['EraPoints'].sum()
        
        # Convert all numeric values to standard Python types
        for index, row in df.iterrows():
            try:
                points_ratio = Decimal(row['EraPoints'] / totalEraPoints) if totalEraPoints > 0 else Decimal(0)
                commission_earned = points_ratio * Decimal(totalRewards) * Decimal(row['CommissionRate'])
                own_reward = (points_ratio * Decimal(totalRewards) - commission_earned) * Decimal(row['OwnStake'] / row['TotalStake']) if row['TotalStake'] > 0 else Decimal(0)
                df.at[index, 'CommissionEarned'] = float(commission_earned)
                df.at[index, 'OwnReward'] = float(own_reward)
                df.at[index, 'TotalReward'] = float(commission_earned + own_reward)
                df.at[index, 'BlocksProduced'] = int(row['EraPoints'] // 20)
                df.at[index, 'EraPoints'] = int(row['EraPoints'])
            except Exception as e:
                logger.warning(f"Error processing row {index}: {str(e)}")
                # Set default values if calculation fails
                df.at[index, 'CommissionEarned'] = 0
                df.at[index, 'OwnReward'] = 0
                df.at[index, 'TotalReward'] = 0
                df.at[index, 'BlocksProduced'] = 0

        df = df.sort_values(by='TotalStake', ascending=False)

        # Format numeric columns with commas
        decimal_columns = ['TotalStake', 'OwnStake', 'NominatedStake', 'CommissionEarned', 'OwnReward', 'TotalReward']
        for col in decimal_columns:
            df[col] = df[col].apply(lambda x: f"{float(x):,.0f}")
        
        df['CommissionRate'] = df['CommissionRate'].apply(lambda x: f"{float(x)*100:.1f}%")
        df['NomCount'] = df['NomCount'].apply(lambda x: int(str(x)))

        columns_order = ["Era", "Validator", "ValidatorName", "ValidatorStash", "EraPoints", "BlocksProduced", "TotalStake", "CommissionRate", "TotalReward"]
        df = df[columns_order + ["NomCount", "OwnStake", "NominatedStake", "CommissionEarned", "OwnReward"]]

        validators_data = df.to_dict('records')
        
        # Remove block history fetching from here
        # Each validator will now have a flag indicating history needs to be loaded
        for validator in validators_data:
            validator['block_history'] = None  # Initialize as None

        # Calculate Staking Ratio and APY
        total_issuance = substrate.query("Balances", "TotalIssuance").value
        total_stake = substrate.query("Staking", "ErasTotalStake", [era]).value
        
        # Staking Ratio calculation
        staking_ratio = (Decimal(total_stake) / Decimal(total_issuance))
        staking_ratio_formatted = f"{float(staking_ratio * 100):.2f}%"
        
        # APY calculation
        era_reward_points = substrate.query("Staking", "ErasRewardPoints", [era]).value
        era_payout = substrate.query("Staking", "ErasValidatorReward", [era - 1]).value
        
        if era_payout:
            annual_reward = Decimal(era_payout) * Decimal(365)  # Assuming 1 era = 1 day
            apy = (annual_reward / Decimal(total_stake)) * Decimal(100)
            apy_formatted = f"{float(apy):.2f}%"
        else:
            apy_formatted = "N/A"

        return validators_data, staking_ratio_formatted, apy_formatted

    except Exception as e:
        logger.error(f"Error processing data for era {era}: {str(e)}")
        return [], "N/A", "N/A"

@app.route('/', methods=['GET', 'POST'])
def index():
    logger.info("Index route accessed")
    try:
        substrate = createInstance()
        logger.info("Substrate instance created")
        reported_current_era = getCurrentEra(substrate)
        logger.info(f"Reported current era: {reported_current_era}")

        selected_era = request.form.get('era', reported_current_era)
        selected_era = int(selected_era)
        logger.info(f"Selected era: {selected_era}")

        validators_data, staking_ratio, apy = processData(substrate , selected_era)
        
        if not validators_data:
            return render_template('index.html', 
                                error_message=f"No data available for era {selected_era}",
                                summary={
                                    'current_era': reported_current_era,
                                    'selected_era': selected_era,
                                    'total_validators': 0,
                                    'total_blocks': 0,
                                    'total_stake': "0",
                                    'staking_ratio': "N/A",
                                    'apy': "N/A"
                                },
                                validators=[])

        # Calculate summary stats
        total_validators = len(validators_data)
        total_blocks = sum(int(v.get('BlocksProduced', 0)) for v in validators_data)
        total_stake = sum(float(v.get('TotalStake', '0').replace(',', '')) for v in validators_data)
        
        summary = {
            'current_era': reported_current_era,
            'selected_era': selected_era,
            'total_validators': total_validators,
            'total_blocks': total_blocks,
            'total_stake': f"{total_stake:,.0f}",
            'staking_ratio': staking_ratio,
            'apy': apy
        }
        
        return render_template('index.html', 
                             summary=summary, 
                             validators=validators_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return render_template('index.html',
                             error_message="An error occurred while fetching data",
                             summary={'current_era': 0, 'selected_era': 0, 
                                    'total_validators': 0, 'total_blocks': 0, 
                                    'total_stake': "0", 'staking_ratio': "N/A", 
                                    'apy': "N/A"},
                             validators=[])

# Add new endpoint for fetching block history
@app.route('/validator_history/<validator_stash>/<int:era>')
def get_validator_block_history(validator_stash, era):
    try:
        substrate = createInstance()
        history = get_validator_history(substrate, validator_stash, era)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching validator history: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
