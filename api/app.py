import os
from pathlib import Path
from flask import Flask, render_template, request
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

def processData(substrate, era):
    df = createDf(substrate, era)
    df = pd.merge(df, getStake(substrate, era), on="Validator", how="inner")
    df = pd.merge(df, validatorPref(substrate, era), on="Validator", how="inner")

    identities = get_validator_identities(substrate, df['Validator'].tolist())

    df['ValidatorName'] = df['Validator'].map(identities)

    logger.info(f"Number of non-empty validator names: {df['ValidatorName'].notna().sum()}")
    logger.info(f"Sample of validator names: {df['ValidatorName'].head().to_dict()}")

    totalRewardsRaw = validatorRewards(substrate, era)
    print(f"Raw total rewards data for era {era}: {totalRewardsRaw}")

    totalRewards = formatBalance(str(totalRewardsRaw)) if totalRewardsRaw is not None else Decimal(0)
    print(f"Total rewards (formatted) for era {era}: {totalRewards}")

    totalEraPoints = df['EraPoints'].sum()
    for index, row in df.iterrows():
        points_ratio = Decimal(row['EraPoints'] / totalEraPoints)
        commission_earned = points_ratio * Decimal(totalRewards) * Decimal(row['CommissionRate'])
        own_reward = (points_ratio * Decimal(totalRewards) - commission_earned) * Decimal(row['OwnStake'] / row['TotalStake'])
        df.at[index, 'CommissionEarned'] = float(commission_earned)
        df.at[index, 'OwnReward'] = float(own_reward)
        df.at[index, 'TotalReward'] = float(commission_earned + own_reward)
        df.at[index, 'BlocksProduced'] = int(row['EraPoints'] / 20)

    df = df.sort_values(by='TotalStake', ascending=False)

    decimal_columns = ['TotalStake', 'OwnStake', 'NominatedStake', 'CommissionEarned', 'OwnReward', 'TotalReward']
    for col in decimal_columns:
        df[col] = df[col].apply(lambda x: f"{float(x):.3f}")
    df['CommissionRate'] = df['CommissionRate'].apply(lambda x: f"{float(x)*100:.1f}%")
    
    df['BlocksProduced'] = df['BlocksProduced'].astype(int)

    columns_order = ["Era", "Validator", "ValidatorName", "EraPoints", "BlocksProduced", "TotalStake", "CommissionRate", "TotalReward"]
    df = df[columns_order + ["NomCount", "OwnStake", "NominatedStake", "CommissionEarned", "OwnReward"]]

    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    logger.info("Index route accessed")
    try:
        substrate = createInstance()
        logger.info("Substrate instance created")
        reported_current_era = getCurrentEra(substrate)
        logger.info(f"Reported current era: {reported_current_era}")

            try:
            df = processData(substrate, reported_current_era)
            current_era = reported_current_era
            except Exception as e:
            logger.warning(f"Data not available for reported era {reported_current_era}. Using previous era.")
            current_era = reported_current_era - 1
            df = processData(substrate, current_era)
        
        logger.info(f"Using era: {current_era}")
        
        selected_era = request.form.get('era', current_era)
        selected_era = int(selected_era)
        logger.info(f"Selected era: {selected_era}")

        if selected_era != current_era:
            df = processData(substrate, selected_era)
        logger.info("Data processed successfully")
        
        total_validators = len(df)
        total_blocks = df['BlocksProduced'].sum()
        total_stake = sum(float(stake) for stake in df['TotalStake'])
        
        summary = {
            'current_era': current_era,
            'selected_era': selected_era,
            'total_validators': total_validators,
            'total_blocks': total_blocks,
            'total_stake': f"{total_stake:.3f} ",
        }
        
        validators_data = df.to_dict('records')
        
        eras = list(range(current_era, max(0, current_era - 10), -1))
        
        logger.info("Rendering template")
        return render_template('index.html', summary=summary, validators=validators_data, eras=eras)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
