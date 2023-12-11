# from glob import glob
# import os
# import shutil
# import pandas as pd
#
# base_dir = '../../data/text'
# event_embeds = 'embedded_umap_10'
# concept_embeds = 'concept_embeds_umap_dim10'
#
# event_dest = 'embedded_umap_10_norm'
# concept_dest = 'concept_embeds_umap_dim10_norm'
#
# def create_folder_or_empty(folder):
#     if os.path.exists(folder):
#         shutil.rmtree(folder)
#     os.mkdir(folder)
#
# if __name__ == "__main__":
#     event_files = glob(f'{base_dir}/{event_embeds}/events-*.pkl')
#     concept_files = glob(f'{base_dir}/{concept_embeds}/concept_*.pkl')
#
#     create_folder_or_empty(f'{base_dir}/{event_dest}')
#     create_folder_or_empty(f'{base_dir}/{concept_dest}')
#
#     for file in event_files:
#         df = pd.read_pickle(file)
#         titles = df['title']
#         summaries = df['summary']
#
#         # make vectors unit length
#         titles = titles.apply(lambda x: np.nor)