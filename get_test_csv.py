import pandas as pd

test_df = pd.read_csv('./data/test.csv', encoding = "ISO-8859-1")
gt_df = pd.read_csv("./data/socialmedia-disaster-tweets-DFE.csv", encoding = "ISO-8859-1")

gt_df = gt_df[['choose_one', 'text']]
gt_df['target'] = (gt_df['choose_one']=='Relevant').astype(int)
gt_df['id'] = gt_df.index
print(gt_df)

merged_df = pd.merge(test_df, gt_df, on='id')

subm_df = merged_df[['id', 'target']]
print(subm_df)

subm_df.to_csv('./data/submission.csv', index=False)


submission = pd.read_csv('./data/submission.csv', encoding="ISO-8859-1")

merged_df = pd.merge(test_df, gt_df, on="id")
merged_df = merged_df[['id', 'target', 'text_x', 'location', 'keyword']]
merged_df.rename(columns={'text_x':'text'}, inplace=True)
print(merged_df)

merged_df.to_csv('./data/test_.csv', encoding="ISO-8859-1", index=False)

# print(subm_df)