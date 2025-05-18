import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🧩 Advanced Data Merge Helper", layout="wide")
st.title("🧩 Advanced Data Merge Helper")

# 1. Upload files (CSV or Excel)
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload first file (CSV or Excel)", type=['csv', 'xlsx'], key="file1")
with col2:
    file2 = st.file_uploader("Upload second file (CSV or Excel)", type=['csv', 'xlsx'], key="file2")

@st.cache_data
def read_file(file):
    if file.name.endswith('.csv'):
        try:
            return pd.read_csv(file)
        except Exception:
            file.seek(0)
            return pd.read_csv(file, encoding='utf-8', errors='ignore')
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)

def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

def calculate_similarity(df1, df2, common_cols):
    similarity_scores = {}
    for col in common_cols:
        vals1 = set(df1[col].dropna().unique())
        vals2 = set(df2[col].dropna().unique())
        intersection = vals1.intersection(vals2)
        union = vals1.union(vals2)
        if len(union) == 0:
            similarity = 0.0
        else:
            similarity = len(intersection) / len(union)
        similarity_scores[col] = similarity
    return similarity_scores

def calculate_composite_key_similarity(df1, df2, cols):
    # Create composite keys by concatenating values as strings with separator
    def composite_key(df, cols):
        return df[cols].astype(str).agg('||'.join, axis=1).dropna().unique()
    keys1 = set(composite_key(df1, cols))
    keys2 = set(composite_key(df2, cols))
    intersection = keys1.intersection(keys2)
    union = keys1.union(keys2)
    if len(union) == 0:
        return 0.0
    else:
        return len(intersection) / len(union)

if file1 and file2:
    df1 = clean_columns(read_file(file1))
    df2 = clean_columns(read_file(file2))

    st.subheader("Preview Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Table 1:** {df1.shape[0]} rows × {df1.shape[1]} cols")
        st.dataframe(df1.head())
    with col2:
        st.markdown(f"**Table 2:** {df2.shape[0]} rows × {df2.shape[1]} cols")
        st.dataframe(df2.head())

    # Detect common columns
    common_cols = sorted(list(set(df1.columns) & set(df2.columns)))
    if not common_cols:
        st.error("No common columns found to merge on.")
    else:
        # Calculate similarity scores
        sim_scores = calculate_similarity(df1, df2, common_cols)
        st.subheader("Similarity Scores for Common Columns")
        sim_df = pd.DataFrame.from_dict(sim_scores, orient='index', columns=['Similarity']).sort_values(by='Similarity', ascending=False)
        st.dataframe(sim_df)

        # Suggest columns with similarity above threshold for merge
        threshold = st.slider("Select similarity threshold to consider columns for merge", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        suggested_cols = sim_df[sim_df['Similarity'] >= threshold].index.tolist()

        st.markdown(f"Columns suggested for merging (similarity ≥ {threshold}):")
        st.write(suggested_cols)

        # User selects columns to merge on (default suggested columns)
        join_cols = st.multiselect("Select one or more columns to merge on", common_cols, default=suggested_cols)

        # If user selected multiple columns, show composite key similarity
        if len(join_cols) > 1:
            comp_sim = calculate_composite_key_similarity(df1, df2, join_cols)
            st.markdown(f"**Composite Key Similarity for selected columns:** {comp_sim:.2f}")

        # Suggest join type (basic heuristic)
        if any('id' in col for col in join_cols):
            suggested_join = 'left'
        else:
            suggested_join = 'inner'

        join_type = st.radio("Select join type", options=['inner', 'left', 'right', 'outer'], index=['inner','left','right','outer'].index(suggested_join), horizontal=True)

        if st.checkbox("Show join types explanation"):
            st.markdown("""
            - **inner**: Only matching rows in both tables  
            - **left**: All rows from first table, matched rows from second  
            - **right**: All rows from second table, matched rows from first  
            - **outer**: All rows from both tables  
            """)

        if st.button("Merge tables"):
            try:
                merged_df = pd.merge(df1, df2, on=join_cols, how=join_type)
                st.success(f"Merged successfully on {join_cols} using {join_type} join.")
                st.write(f"Result: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
                st.dataframe(merged_df.head(50))

                # Plot comparison
                st.subheader("Data Size Comparison")
                sizes = {
                    "Table 1": df1.shape[0],
                    "Table 2": df2.shape[0],
                    "Merged": merged_df.shape[0]
                }
                fig, ax = plt.subplots()
                ax.bar(sizes.keys(), sizes.values(), color=['blue','orange','green'])
                ax.set_ylabel('Number of rows')
                st.pyplot(fig)

                # Download button
                csv = merged_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download merged CSV", csv, "merged.csv", "text/csv")

                # Show merge code
                st.markdown("### Merge Code")
                cols_str = ', '.join([f'"{c}"' for c in join_cols])
                st.code(f'pd.merge(df1, df2, on=[{cols_str}], how="{join_type}")', language='python')
            except Exception as e:
                st.error(f"Error during merge: {e}")
