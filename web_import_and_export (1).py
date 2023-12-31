# -*- coding: utf-8 -*-
"""web import and export.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ejW5z_x6WY7EW6Yh4QETV7MrVai_fSLb
"""

!pip install ipython-autotime pandas bs4

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autotime

"""### Import libraries"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

"""### Define country codes and base options"""

country_list = {'AFGHANISTAN': '1',
                'ALBANIA': '3',
                'ALGERIA': '5',
                'AMERI SAMOA': '7',
                'ANDORRA': '9',
                'ANGOLA': '11',
                'ANGUILLA': '12',
                'ANTARTICA': '14',
                'ANTIGUA': '13',
                'ARGENTINA': '15',
                'ARMENIA': '16',
                'ARUBA': '20',
                'AUSTRALIA': '17',
                'AUSTRIA': '19',
                'AZERBAIJAN': '21',
                'BAHAMAS': '23',
                'BAHARAIN IS': '25',
                'BANGLADESH PR': '27',
                'BARBADOS': '29',
                'BELARUS': '55',
                'BELGIUM': '33',
                'BELIZE': '31',
                'BENIN': '35',
                'BERMUDA': '37',
                'BHUTAN': '38',
                'BOLIVIA': '39',
                'BOSNIA-HRZGOVIN': '40',
                'BOTSWANA': '41',
                'BR VIRGN IS': '45',
                'BRAZIL': '43',
                'BRUNEI': '47',
                'BULGARIA': '49',
                'BURKINA FASO': '50',
                'BURUNDI': '53',
                'C AFRI REP': '67',
                'CAMBODIA': '56',
                'CAMEROON': '57',
                'CANADA': '59',
                'CANARY IS': '61',
                'CAPE VERDE IS': '63',
                'CAYMAN IS': '65',
                'CHAD': '69',
                'CHANNEL IS': '71',
                'CHILE': '73',
                'CHINA P RP': '77',
                'CHRISTMAS IS.': '79',
                'COCOS IS': '81',
                'COLOMBIA': '83',
                'COMOROS': '85',
                'CONGO D. REP.': '459',
                'CONGO P REP': '87',
                'COOK IS': '89',
                'COSTA RICA': '91',
                "COTE D' IVOIRE": '199',
                'CROATIA': '92',
                'CUBA': '93',
                'CURACAO': '276',
                'CYPRUS': '95',
                'CZECH REPUBLIC': '98',
                'DENMARK': '101',
                'DJIBOUTI': '102',
                'DOMINIC REP': '103',
                'DOMINICA': '105',
                'ECUADOR': '109',
                'EGYPT A RP': '111',
                'EL SALVADOR': '113',
                'EQUTL GUINEA': '117',
                'ERITREA': '116',
                'ESTONIA': '114',
                'ETHIOPIA': '115',
                'FALKLAND IS': '123',
                'FAROE IS.': '121',
                'FIJI IS': '127',
                'FINLAND': '125',
                'FR GUIANA': '131',
                'FR POLYNESIA': '133',
                'FR S ANT TR': '135',
                'FRANCE': '129',
                'GABON': '141',
                'GAMBIA': '143',
                'GEORGIA': '145',
                'GERMANY': '147',
                'GHANA': '149',
                'GIBRALTAR': '151',
                'GREECE': '155',
                'GREENLAND': '157',
                'GRENADA': '159',
                'GUADELOUPE': '161',
                'GUAM': '163',
                'GUATEMALA': '165',
                'GUERNSEY': '124',
                'GUINEA': '167',
                'GUINEA BISSAU': '169',
                'GUYANA': '171',
                'HAITI': '175',
                'HEARD MACDONALD': '176',
                'HONDURAS': '177',
                'HONG KONG': '179',
                'HUNGARY': '181',
                'ICELAND': '185',
                 'INDONESIA': '187',
                'INSTALLATIONS IN INTERNATIONAL WATERS   ': '2',
                'IRAN': '189',
                'IRAQ': '191',
                'IRELAND': '193',
                'ISRAEL': '195',
                'ITALY': '197',
                'JAMAICA': '203',
                'JAPAN': '205',
                'JERSEY         ': '122',
                'JORDAN': '207',
                'KAZAKHSTAN': '212',
                'KENYA': '213',
                'KIRIBATI REP': '214',
                'KOREA DP RP': '215',
                'KOREA RP': '217',
                'KUWAIT': '219',
                'KYRGHYZSTAN': '216',
                'LAO PD RP': '223',
                'LATVIA': '224',
                'LEBANON': '225',
                'LESOTHO': '227',
                'LIBERIA': '229',
                'LIBYA': '231',
                'LIECHTENSTEIN': '233',
                'LITHUANIA': '234',
                'LUXEMBOURG': '235',
                'MACAO': '239',
                'MACEDONIA': '240',
                 'MADAGASCAR': '241',
                'MALAWI': '243',
                'MALAYSIA': '245',
                'MALDIVES': '247',
                'MALI': '249',
                'MALTA': '251',
                'MARSHALL ISLAND': '252',
                'MARTINIQUE': '253',
                'MAURITANIA': '255',
                'MAURITIUS': '257',
                'MAYOTTE': '34',
                'MEXICO': '259',
                'MICRONESIA': '256',
                'MOLDOVA': '260',
                'MONACO': '262',
                'MONGOLIA': '261',
                'MONTENEGRO': '356',
                'MONTSERRAT': '263',
                'MOROCCO': '265',
                'MOZAMBIQUE': '267',
                'MYANMAR': '258',
                'N. MARIANA IS.': '294',
                'NAMIBIA': '269',
                'NAURU RP': '271',
                'NEPAL': '273',
                'NETHERLAND': '275',
                'NETHERLANDANTIL': '277',
                'NEUTRAL ZONE': '279',
                'NEW CALEDONIA': '281',
                'NEW ZEALAND': '285',
                'NICARAGUA': '287',
                'NIGER': '289',
                'NIGERIA': '291',
                'NIUE IS': '293',
                'NORFOLK IS': '295',
                'NORWAY': '297',
                'OMAN': '301',
                'PACIFIC IS': '307',
                'PAKISTAN IR': '309',
                'PALAU': '310',
                'PANAMA C Z': '313',
                'PANAMA REPUBLIC': '311',
                'PAPUA N GNA': '315',
                'PARAGUAY': '317',
                'PERU': '319',
                'Petroleum Products': '0',
                'PHILIPPINES': '323',
                'PITCAIRN IS.': '321',
                'POLAND': '325',
                'PORTUGAL': '327',
                'PUERTO RICO': '331',
                'QATAR': '335',
                'REUNION': '339',
                'ROMANIA': '343',
                'RUSSIA': '344',
                'RWANDA': '345',
                 'SAHARWI A.DM RP': '347',
                'SAMOA': '447',
                'SAN MARINO': '346',
                'SAO TOME': '349',
                'SAUDI ARAB': '351',
                'SENEGAL': '353',
                'SERBIA': '352',
                'SEYCHELLES': '355',
                'SIERRA LEONE': '357',
                'SINGAPORE': '359',
                'SINT MAARTEN (DUTCH PART)': '278',
                'SLOVAK REP': '358',
                'SLOVENIA': '360',
                'SOLOMON IS': '361',
                'SOMALIA': '363',
                'SOUTH AFRICA': '365',
                'SOUTH SUDAN ': '382',
                'SPAIN': '367',
                'SRI LANKA DSR': '369',
                'ST HELENA': '371',
                'ST KITT N A': '373',
                'ST LUCIA': '375',
                'ST PIERRE': '377',
                'ST VINCENT': '379',
                'STATE OF PALEST': '196',
                'SUDAN': '381',
                 'SURINAME': '383',
                'SVALLBARD AND J': '6',
                'SWAZILAND': '385',
                'SWEDEN': '387',
                'SWITZERLAND': '389',
                'SYRIA': '391',
                'TAIWAN': '75',
                'TAJIKISTAN': '393',
                'TANZANIA REP': '395',
                'THAILAND': '397',
                'TIMOR LESTE': '329',
                'TOGO': '399',
                'TOKELAU IS': '401',
                'TONGA': '403',
                'Trade to Unspecified Countries': '999',
                'TRINIDAD': '405',
                'TUNISIA': '407',
                'TURKEY': '409',
                'TURKMENISTAN': '410',
                'TURKS C IS': '411',
                'TUVALU': '413',
                'U ARAB EMTS': '419',
                'U K': '421',
                'U S A': '423',
                'UGANDA': '417',
                'UKRAINE': '422',
                'UNION OF SERBIA & MONTENEGRO': '354',
                'UNSPECIFIED': '599',
                'URUGUAY': '427',
                'US MINOR OUTLYING ISLANDS               ': '424',
                'UZBEKISTAN': '430',
                'VANUATU REP': '431',
                 'VATICAN CITY': '198',
                'VENEZUELA': '433',
                'VIETNAM SOC REP': '437',
                'VIRGIN IS US': '439',
                'WALLIS F IS': '443',
                'YEMEN REPUBLC': '453',
                'ZAMBIA': '461',
                'ZIMBABWE': '463'}

urls = {
    'export': "https://tradestat.commerce.gov.in/eidb/ecntcom.asp",
    'import': "https://tradestat.commerce.gov.in/eidb/icntcom.asp"
}
options = {
    'hslevel': 2,
    'sort': 0,
    'radioDAll': 1,
    'radiousd': 1
}

session = requests.Session()

"""### Function to get trade data for a specified country in a specified year"""

def get_trade_data(req_type='export', year=2024, country='U S A'):
    url = urls[req_type]

    options['yy1'] = year
    options['cntcode'] = country_list[country]

    sample = session.post(url, options)
    try:
        # remove extra rows from bottom
        response_df = pd.read_html(sample.content)[0].iloc[:-3, 1:5]
        # remove previous year data
        response_df = response_df.drop(
            response_df.columns[2], axis=1)
        # rename to "value"
        response_df.rename(
            columns={response_df.columns[2]: 'value'}, inplace=True)
        response_df['country'] = country
        response_df['year'] = year
        return response_df

    except:
        print(f"Unable to extract: {country} for {year}")
        pass

"""###  Function to get trade data (export / import) for all years and countries"""

def get_data(req_type='export'):
    dfs = []
    for year in range(2010, 2024):
        for country in list(country_list.keys()):
            print(f"Extracting {country} - {year}")
            dfs.append(get_trade_data(req_type, year, country))

    return pd.concat(dfs)

"""###Get export data"""

export_data = get_data('export')
export_data.to_csv('./2010_2023_export_data.csv', index=False)

"""### Get import data"""

import_data = get_data('import')
import_data.to_csv('./2010_2023_import_data.csv', index=False)

export_data

import_data