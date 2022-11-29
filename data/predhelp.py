def dict_clean (px_data_url, regions):
    px = rf'{px_data_url}'
    px = pyaxis.parse(uri = px , encoding = 'ISO-8859-2')
    df = px['DATA']
    dict_meta = px['METADATA']
    bad_countries = ['Übriges Südamerika',
                     'Übriges Zentralamerika, Karibik',
                     'Übriges Nordafrika',
                     'Übriges Afrika',
                     'Übriges Westasien',
                     'Übriges Süd- und Ostasien',
                     'Übriges Europa']
    for m in range(0, len(bad_countries)):
        df = df.loc[df['Herkunftsland'] != bad_countries[m]]
    df = df.loc[(df['Indikator'] == 'Logiernächte') & (df['DATA'] != '"..."') & (df['DATA'] != '"......"')]
    df["Jahr"] = df["Jahr"].astype(int)
    df["DATA"] = df["DATA"].astype(int)
    df_months = df.loc[(df['Monat'] != 'Jahrestotal')]
    df_months = df_months.groupby('Gemeinde')
    all_regions = df_months['Gemeinde'].unique()
    dict_regions = [{} for k in range(0, len(regions))]
    for i in range(0, len(regions)):
        dict_regions[i][regions[i]] = df_months.get_group(regions[i])
        dict_regions[i][regions[i]].reset_index(inplace = True)
        dict_regions[i][regions[i]].rename(columns = {'Jahr' : 'year', 'Monat': 'month'}, inplace = True)
    trans_dict = {
        'Schweiz' : 'Switzerland',
        'Deutschland' : 'Germany',
        'Italien': 'Italy',
        'Frankreich' : 'France',
        'Österreich' : 'Austria',
        'Vereinigtes Königreich': 'United Kingdom',
        'Irland' : 'Ireland',
        'Niederlande' : 'Netherlands',
        'Belgien' : 'Belgium',
        'Luxemburg' : 'Luxembourg',
        'Dänemark' : 'Denmark',
        'Schweden' : 'Sweden',
        'Norwegen' : 'Norway',
        'Finnland' : 'Finland',
        'Spanien' : 'Spain',
        'Griechenland' : 'Greece',
        'Türkei' : 'Turkey',
        'Island' : 'Iceland',
        'Polen' : 'Polony',
        'Ungarn' : 'Ungary',
        'Bulgarien': 'Bulgary',
        'Zypern' : 'Cyprus',
        'Vereinigte Staaten' : 'United States',
        'Kanada' : 'Canada',
        'Mexiko' : 'Mexico',
        'Brasilien' : 'Brasil',
        'Portugal' : 'Portugal',
        'Argentinien' : 'Argentine',
        'Ägypten' : 'Egypt',
        'Südafrika' : 'South Africa',
        'Indien' : 'India',
        'Katar' : 'Qatar',
        'Asutralien' : 'Australia',
        'Indonesien' : 'Indonesia',
        'Korea (Süd-)' : 'South Korea',
        'Phillipinen' : 'Philippines',
        'Neuseeland, Ozeanien' : 'New Zealand',
        'Singapur' : 'Singapore',
        'Taiwan (Chinesisches Taipei)' : 'Taiwan',
        'Estland' : 'Estonia',
        'Lettland' : 'Latvia',
        'Litauen' : 'Lithuania',
        'Saudi-Arabien' : 'Saudi Arabia',
        'Vereinigte Arabische Emirate' : 'United Arab Emirates',
        'Kroatien' : 'Croatia',
        'Rumänien' : 'Romania',
        'Russland' : 'Russia',
        'Slowakei' : 'Slovakia',
        'Slowenien' : 'Slovenia',
        'Tschechien' : 'Czech Republic',
        'Serbien' : 'Serbia',
        }
    
    for k in range(0, len(dict_regions)):
        for i in range(0, len(dict_regions[k][regions[k]])):
            if dict_regions[k][regions[k]].iloc[i]['Herkunftsland'] in trans_dict.keys():
                dict_regions[k][regions[k]].at[i,'Herkunftsland'] = trans_dict[dict_regions[k][regions[k]].iloc[i]['Herkunftsland']]
    trans_months = {
        'Januar' : 1,
        'Februar' : 2,
        'März' : 3,
        'April' : 4,
        'Mai' : 5,
        'Juni' : 6,
        'Juli' : 7,
        'August' : 8,
        'September' : 9,
        'Oktober' : 10,
        'November' : 11,
        'Dezember' : 12}
    for k in range(0, len(dict_regions)):
        for i in range(0, len(dict_regions[k][regions[k]])):
            if dict_regions[k][regions[k]].iloc[i]['month'] in trans_months.keys():
                dict_regions[k][regions[k]].at[i,'month'] = trans_months[dict_regions[k][regions[k]].iloc[i]['month']]
    for i in range(0, len(dict_regions)):
        dict_regions[i][regions[i]]['date'] = pd.to_datetime(dict_regions[i][regions[i]][['year','month']].assign(DAY = 1))
        dict_regions[i][regions[i]].drop(labels = ['year', 'month'], axis = 1, inplace = True)
    for i in range(0, len(dict_regions)):
        dict_regions[i][regions[i]] = dict_regions[i][regions[i]].pivot(index =  ['date'],columns = 'Herkunftsland', values = 'DATA').reset_index().rename_axis(None, axis=1)
     
    return(dict_regions, all_regions)