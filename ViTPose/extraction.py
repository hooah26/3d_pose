import pandas as pd
from collections import Counter
import copy
from tqdm import tqdm
import numpy as np
from glob import glob
from collections import defaultdict


def split_thd(rm_flag=True):
    df = pd.read_csv('json_info_updated.csv')
    print(df)
    # print(f"0.0: {len(df.loc[df['oks'] == 0.0])}, -1.0: {len(df.loc[df['oks'] == -1.0])}")
    # print(df.shape[0])
    thd = 30.0

    if rm_flag:
        df_rm_list = pd.read_csv('missing_kpt_img.csv')
        rm_list = []
        for i in tqdm(range(len(df_rm_list))):
            rm_list.append('_'.join(df_rm_list.loc[i, 'img_path'].split('/')[-1].split('_')[0:2]))
        rm_list.append('53_F170A')
        rm_list.append('53_F170D')

        rm_list.append('13_M170B')
        rm_list.append('44_F160C')
        rm_list.append('13_F160A')
        rm_list.append('37_F170D')
        rm_list.append('44_M170A')
        rm_list.append('32_M170B')
        rm_list.append('39_M160C')
        rm_list.append('40_M160D')
        rm_list.append('19_F150C')


        split_thd_over(df, thd, set(rm_list))
    else:
        split_thd_over(df, thd)


def split_thd_over(df, thd, rm_list=None):
    df_thd_over = copy.deepcopy(df[df['oks'] >= thd].sort_values('oks', ascending=False)).reset_index(drop=True)
    df_thd_over.columns = ['json_path', 'oks']
    print(df_thd_over)

    for i in tqdm(range(len(df_thd_over))):
        df_thd_over.loc[i, 'SA'] = '_'.join(df_thd_over.loc[i, 'json_path'].split('/')[-1].split('_')[0:2])
        df_thd_over.loc[i, 'SAC'] = '_'.join(df_thd_over.loc[i, 'json_path'].split('/')[-1].split('_')[0:3])

    if rm_list is not None:
        idx = df_thd_over[df_thd_over['SA'].isin(list(set(rm_list)))].index
        df_thd_over.drop(idx, inplace=True)

    df_thd_over.reset_index(drop=True, inplace=True)
    print(df_thd_over)
    df_thd_over.to_csv('thd_over.csv', index=False)

    dict_thd_over_sa = defaultdict(lambda: [0, 0.0])
    dict_thd_over_sac = defaultdict(lambda: [0, 0.0])
    for i in tqdm(range(len(df_thd_over))):
        dict_thd_over_sa[df_thd_over.loc[i, 'SA']][0] += 1
        dict_thd_over_sa[df_thd_over.loc[i, 'SA']][1] += df_thd_over.loc[i, 'oks']
        dict_thd_over_sac[df_thd_over.loc[i, 'SAC']][0] += 1
        dict_thd_over_sac[df_thd_over.loc[i, 'SAC']][1] += df_thd_over.loc[i, 'oks']

    dict_thd_over_sa = {key: value[1] / value[0] for key, value in dict_thd_over_sa.items()}
    dict_thd_over_sa_sorted = pd.DataFrame(dict_thd_over_sa.items(), columns=['SA', 'oks_average']).sort_values('oks_average', ascending=False).reset_index(drop=True)
    print(dict_thd_over_sa_sorted)
    dict_thd_over_sa_sorted.to_csv('thd_over_SA.csv', index=False)

    dict_thd_over_sac = {key: value[1] / value[0] for key, value in dict_thd_over_sac.items()}
    dict_thd_over_sac_sorted = pd.DataFrame(dict_thd_over_sac.items(), columns=['SAC', 'oks_average']).sort_values('oks_average', ascending=False).reset_index(drop=True)
    print(dict_thd_over_sac_sorted)
    dict_thd_over_sac_sorted.to_csv('thd_over_SAC.csv', index=False)

    split_thd_under(df, thd, dict_thd_over_sa_sorted, rm_list,)


def split_thd_under(df, thd, thd_over=None, rm_list=None):
    df_thd_under = copy.deepcopy(df[df['oks'] < thd].sort_values('oks', ascending=False)).reset_index(drop=True)#[0:int(len(df)/1000)]
    df_thd_under.columns = ['json_path', 'oks']
    print(df_thd_under)


    for i in tqdm(range(len(df_thd_under))):
        df_thd_under.loc[i, 'SA'] = '_'.join(df_thd_under.loc[i, 'json_path'].split('/')[-1].split('_')[0:2])
        df_thd_under.loc[i, 'SAC'] = '_'.join(df_thd_under.loc[i, 'json_path'].split('/')[-1].split('_')[0:3])

    if rm_list is not None:
        idx = df_thd_under[df_thd_under['SA'].isin(list(set(rm_list)))].index
        df_thd_under.drop(idx, inplace=True)

    if thd_over is not None:
        idx = df_thd_under[df_thd_under['SA'].isin(thd_over["SA"])].index
        df_thd_under.drop(idx, inplace=True)

    df_thd_under.reset_index(drop=True, inplace=True)
    print(df_thd_under)
    df_thd_under.to_csv('thd_under.csv', index=False)

    df_thd_under_sa = defaultdict(lambda: [0, 0.0])
    df_thd_under_sac = defaultdict(lambda: [0, 0.0])
    for i in tqdm(range(len(df_thd_under))):
        df_thd_under_sa[df_thd_under.loc[i, 'SA']][0] += 1
        df_thd_under_sa[df_thd_under.loc[i, 'SA']][1] += df_thd_under.loc[i, 'oks']
        df_thd_under_sac[df_thd_under.loc[i, 'SAC']][0] += 1
        df_thd_under_sac[df_thd_under.loc[i, 'SAC']][1] += df_thd_under.loc[i, 'oks']

    df_thd_under_sa = {key: value[1] / value[0] for key, value in df_thd_under_sa.items()}
    df_thd_under_sa_sorted = pd.DataFrame(df_thd_under_sa.items(), columns=['SA', 'oks_average']).sort_values('oks_average', ascending=False).reset_index(drop=True)
    print(df_thd_under_sa_sorted)
    df_thd_under_sa_sorted.to_csv('thd_under_SA.csv', index=False)

    df_thd_under_sac = {key: value[1] / value[0] for key, value in df_thd_under_sac.items()}
    df_thd_under_sac_sorted = pd.DataFrame(df_thd_under_sac.items(), columns=['SAC', 'oks_average']).sort_values('oks_average', ascending=False).reset_index(drop=True)
    print(df_thd_under_sac_sorted)
    df_thd_under_sac_sorted.to_csv('thd_under_SAC.csv', index=False)


def sort_thd_under():
    thd_under = pd.read_csv('thd_under.csv')
    thd_under_sa = pd.read_csv('thd_under_SA.csv')
    thd_under_sac = pd.read_csv('thd_under_SAC.csv')

    print(thd_under)
    print(thd_under_sa)
    print(thd_under_sac)

    # 'SA_rank' 열 초기화
    thd_under['SA_rank'] = None
    thd_under['SAC_rank'] = None

    # thd_under과 thd_under_SA를 비교하여 일치하는 경우 인덱스 할당
    for index, row in tqdm(thd_under.iterrows()):
        matching_sa = thd_under_sa.index[thd_under_sa['SA'] == row['SA']]
        if len(matching_sa) > 0:
            thd_under.at[index, 'SA_rank'] = matching_sa[0]

        matching_sac = thd_under_sac.index[thd_under_sac['SAC'] == row['SAC']]
        if len(matching_sac) > 0:
            thd_under.at[index, 'SAC_rank'] = matching_sac[0]

    # 결과 확인
    thd_under_sorted = thd_under.sort_values(by=["SA_rank", "SAC_rank"], ascending=[True, True]).reset_index(drop=True)
    print(thd_under_sorted)
    thd_under_sorted.to_csv('thd_under_sorted.csv', index=False)


def simply_sort():
    thd_under_sorted = pd.read_csv('thd_under_sorted.csv')
    print(thd_under_sorted)

    df_thd_under_sorted = defaultdict(lambda: ["", 0, 0, 0])
    for i in tqdm(range(len(thd_under_sorted))):
        df_thd_under_sorted[thd_under_sorted.loc[i, 'SAC']][0] = thd_under_sorted.loc[i, 'SA']
        df_thd_under_sorted[thd_under_sorted.loc[i, 'SAC']][1] += 1
        df_thd_under_sorted[thd_under_sorted.loc[i, 'SAC']][2] = thd_under_sorted.loc[i, 'SA_rank']
        df_thd_under_sorted[thd_under_sorted.loc[i, 'SAC']][3] = thd_under_sorted.loc[i, 'SAC_rank']

    # print(df_thd_under_sorted)

    # df_simple = pd.DataFrame.from_dict(df_thd_under_sorted, orient='index')
    df_simple = pd.DataFrame.from_dict(df_thd_under_sorted, orient='index', columns=['SA', 'cnt', 'SA_rank','SAC_rank']).reset_index().rename(columns={'index': 'SAC'})
    df_simple = df_simple[['SA', 'SAC', 'cnt', 'SA_rank', 'SAC_rank']]
    df_simple = df_simple.sort_values(by=["SA_rank", "SAC_rank"], ascending=[True, True]).reset_index(drop=True)
    print(df_simple)
    df_simple.to_csv('thd_under_sorted_simple.csv', index=False)


def ext_10per():
    df = pd.read_csv('thd_under_sorted_simple.csv')
    print(df)

    df_tenper = pd.DataFrame([], columns=['SA', 'SAC', 'cnt', 'SA_rank', 'SAC_rank'])
    count = 0

    for key, group in df.groupby("SA_rank"):
        if count == 100:
            break
        if len(group) > 2:
            df_tenper = pd.concat([df_tenper, group], ignore_index=True)
            count += 1
    df_tenper.reset_index(drop=True)
    print(df_tenper)
    df_tenper.to_csv('ext_10per.csv', index=False)

    df_tenper_3cam = df.groupby("SA")[["SA", "SAC","cnt", "SA_rank", "SAC_rank"]].apply(lambda x: x.head(n=3 if len(x) > 2 else 0)).sort_values(by=["SA_rank", "SAC_rank"]).head(300).droplevel(level=0)
    print(df_tenper_3cam)
    df_tenper_3cam.to_csv('ext_10per_3cam.csv', index=False)


def main():
    split_thd(True)
    sort_thd_under()
    simply_sort()
    ext_10per()
    # df = pd.read_csv('thd_under_sorted.csv')
    #
    # df = pd.read_csv('ext_10per.csv')
    # print(df)



if __name__ == '__main__':
    main()
