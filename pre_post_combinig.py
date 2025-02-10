import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import ttest_rel, t

# Title
st.title("Survey Data Visualization Dashboard")
st.info("Please ensure that the files have only one header row.", icon="ℹ️")

# File Upload
st.subheader("Upload Pre and Post Survey Data")
pre_file = st.file_uploader(
    "Upload Pre-Survey Data (CSV/Excel)", type=["csv", "xls", "xlsx"])
post_file = st.file_uploader(
    "Upload Post-Survey Data (CSV/Excel)", type=["csv", "xls", "xlsx"])


@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, dtype=str) if file.name.endswith(".csv") else pd.read_excel(file, dtype=str)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


if pre_file and post_file:
    pre_df, post_df = load_data(pre_file), load_data(post_file)

    if pre_df is not None and post_df is not None:
        st.write("Pre-Survey Data Preview:")
        st.dataframe(pre_df.head())

        st.write("Post-Survey Data Preview:")
        st.dataframe(post_df.head())

        # Column Selection for Merging
        st.subheader("Select Columns to Merge On")
        pre_merge_col = st.selectbox(
            "Select Pre-Survey Column", pre_df.columns, key="pre_col")
        post_merge_col = st.selectbox(
            "Select Post-Survey Column", post_df.columns, key="post_col")

        merge_type = st.radio("Select Merge Type", [
                              "inner", "outer", "left", "right"])

        # Standardize Merge Columns
        if st.checkbox("Standardize Merge Columns (Trim spaces & Convert to lowercase)", value=True):
            pre_df[pre_merge_col] = pre_df[pre_merge_col].str.strip().str.lower()
            post_df[post_merge_col] = post_df[post_merge_col].str.strip().str.lower()

        # Merging
        if st.button("Merge Data"):
            merged_df = pd.merge(pre_df, post_df, left_on=pre_merge_col, right_on=post_merge_col,
                                 suffixes=("_pre", "_post"), how=merge_type)

            # Store merged dataframe in session state
            st.session_state["merged_df"] = merged_df
            st.success(f"✅ Merging Successful! {len(merged_df)} rows merged.")
            st.dataframe(merged_df.head())

            # Identify unmatched rows in both datasets
            pre_unmatched = pre_df[~pre_df[pre_merge_col].isin(
                merged_df[pre_merge_col])][[pre_merge_col]]
            post_unmatched = post_df[~post_df[post_merge_col].isin(
                merged_df[post_merge_col])][[post_merge_col]]

            # Display only merge column values by default
            if not pre_unmatched.empty:
                st.warning(f"{len(
                    pre_unmatched)} rows from Pre-Survey could not be merged (No match in Post-Survey).")
                st.dataframe(pre_unmatched, hide_index=True)

                # Option to expand and view full rows
                if st.checkbox("Show full unmatched Pre-Survey rows"):
                    st.dataframe(
                        pre_df[~pre_df[pre_merge_col].isin(merged_df[pre_merge_col])])

            if not post_unmatched.empty:
                st.warning(f"{len(
                    post_unmatched)} rows from Post-Survey could not be merged (No match in Pre-Survey).")
                st.dataframe(post_unmatched, hide_index=True)

                # Option to expand and view full rows
                if st.checkbox("Show full unmatched Post-Survey rows"):
                    st.dataframe(
                        post_df[~post_df[post_merge_col].isin(merged_df[post_merge_col])])

        # Data Preprocessing
        if "merged_df" in st.session_state:
            merged_df = st.session_state["merged_df"]

            st.subheader("🔧 Data Preprocessing")

            # Drop Duplicates
            if st.checkbox("Drop Duplicate Rows"):
                if st.button("Confirm Remove Duplicates"):
                    merged_df.drop_duplicates(inplace=True)
                    st.success("✅ Duplicate rows removed.")

            # Handle Missing Values
            missing_cols = merged_df.columns[merged_df.isna().any()].tolist()
            if missing_cols:
                st.warning(f"⚠️ Missing values detected")
                selected_cols = st.multiselect(
                    "Select columns to handle missing values:", missing_cols)

                fill_method = st.selectbox("Choose Fill Method", [
                                           "Drop Rows", "Fill with Mean", "Fill with Mode", "Custom Value"])
                if fill_method == "Drop Rows" and st.button("Confirm Drop"):
                    merged_df.dropna(subset=selected_cols, inplace=True)
                    st.success("🚀 Rows with missing values dropped.")
                elif fill_method == "Fill with Mean":
                    for col in selected_cols:
                        if pd.api.types.is_numeric_dtype(merged_df[col]):
                            merged_df[col].fillna(
                                merged_df[col].astype(float).mean(), inplace=True)
                    st.success("✅ Missing values filled with mean.")
                elif fill_method == "Fill with Mode":
                    for col in selected_cols:
                        merged_df[col].fillna(
                            merged_df[col].mode()[0], inplace=True)
                    st.success("✅ Missing values filled with mode.")
                elif fill_method == "Custom Value":
                    custom_value = st.text_input("Enter custom fill value")
                    if st.button("Apply Custom Fill"):
                        merged_df[selected_cols] = merged_df[selected_cols].fillna(
                            custom_value)
                        st.success(f"✅ Missing values filled with '{
                                   custom_value}'")

            # Likert Scale Mapping
            st.subheader("Likert Scale Mapping")
            likert_scales = {
                "scale_1": {"Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2, "Strongly Disagree": 1},
                "scale_2": {"Not Confident At All": 1, "Not Very Confident": 2, "Neutral": 3, "Moderately Confident": 4, "Very Confident": 5},
                "scale_3": {"Strongly Disagree": 1, "Disagree": 2, "Somewhat Disagree": 3, "Somewhat Agree": 4, "Agree": 5, "Strongly Agree": 6},
                "scale_4": {"Not at all prepared": 1, "Slightly prepared": 2, "Prepared": 3, "Very prepared": 4, "Extremely prepared": 5},
                "scale_5": {"Not at all confident": 1, "Slightly confident": 2, "Confident": 3, "Very confident": 4, "Extremely confident": 5}
            }

            def detect_likert_scale(column_values):
                for scale_name, scale_map in likert_scales.items():
                    if set(column_values.dropna()).issubset(set(scale_map.keys())):
                        return scale_name, scale_map
                return None, None

            # Detect Likert-compatible columns
            text_columns = merged_df.select_dtypes(include=["object"]).columns
            likert_columns = [
                col for col in text_columns if detect_likert_scale(merged_df[col])[0]]

            # Preselect all Likert-compatible columns by default
            selected_columns = st.multiselect(
                "Select Likert scale columns", likert_columns, default=likert_columns)

            # Apply Likert scale mapping
            if st.button("Map Likert Scales"):
                failed_columns = []
                mapped_columns = []

                for col in selected_columns:
                    _, scale_map = detect_likert_scale(merged_df[col])
                    if scale_map:
                        new_col = col + "_mapped"
                        merged_df[new_col] = merged_df[col].map(scale_map)
                        mapped_columns.append(new_col)
                    else:
                        failed_columns.append(col)

                # Show success message if at least one column was mapped
                if mapped_columns:
                    st.success("✅ Likert scale mapping applied successfully!")

                    # Show a preview of mapped columns
                    st.subheader("🔍 Mapped Columns Preview")
                    st.dataframe(merged_df[mapped_columns].head())

                # Show error messages for columns that failed
                for col in failed_columns:
                    st.error(f"❌ Could not map column: {
                             col} - No matching Likert scale found.")

            # Store updates in session state
            st.session_state["merged_df"] = merged_df

            st.subheader("📊 Pre-Post Survey Comparison")
            if "merged_df" in st.session_state:
                merged_df = st.session_state["merged_df"]
                numeric_cols = [col for col in merged_df.columns if merged_df[col].dtype in [
                    'int64', 'float64']]

                # Select Pre-Post Columns
                selected_pre_col = st.selectbox(
                    # if col.endswith("_pre_mapped")])
                    "Select Pre-Survey Column", [col for col in numeric_cols if col.endswith("_pre_mapped") or col.endswith("_pre")])
                selected_post_col = None

                if selected_pre_col:
                    selected_post_col = selected_pre_col.replace(
                        "_pre", "_post") if selected_pre_col.replace("_pre", "_post") in numeric_cols else None

                if selected_post_col:
                    # Filter & Grouping Options
                    filter_col = st.selectbox("Filter Data By", [
                                              None] + list(merged_df.select_dtypes(include=['object']).columns))
                    filter_value = st.selectbox("Select Filter Value", merged_df[filter_col].dropna(
                    ).unique()) if filter_col else None
                    filtered_df = merged_df[merged_df[filter_col]
                                            == filter_value] if filter_col else merged_df

                    group_col = st.selectbox(
                        "Group By", [None] + list(merged_df.select_dtypes(include=['object']).columns))
                    grouped_df = filtered_df.groupby(group_col)[[selected_pre_col, selected_post_col]].mean().reset_index(
                    ) if group_col else filtered_df[[selected_pre_col, selected_post_col]].mean().to_frame().T
                    grouped_df.insert(
                        0, 'Overall', 'Overall') if not group_col else None

                    # Compute Percentage Change
                    grouped_df['Percentage Change'] = (
                        (grouped_df[selected_post_col] - grouped_df[selected_pre_col]) / grouped_df[selected_pre_col]) * 100

                    # Ensure `group_col` is not None
                    x_axis_title = group_col if group_col else "Overall"

                    # If grouping is not used, provide a placeholder category for tick labels
                    tick_labels = grouped_df[group_col].astype(
                        str).tolist() if group_col else ["Overall"]

                    # Truncate long labels
                    tick_labels = [
                        label[:10] + "..." if len(label) > 10 else label for label in tick_labels]

                    # Visualization - Pre vs Post
                    fig = px.bar(
                        grouped_df, x=group_col, y=[
                            selected_pre_col, selected_post_col],
                        barmode='group', title=f"Comparison of {selected_pre_col} and {selected_post_col}",
                        labels={selected_pre_col: "Pre-Survey",
                                selected_post_col: "Post-Survey"},
                        # Ensure full category names appear in hover
                        # hover_data={group_col: True}
                    )
                    # Improve readability
                    fig.update_layout(
                        template="plotly_white",
                        legend_title="Survey Type",
                        legend_traceorder="normal",
                        xaxis=dict(
                            title=x_axis_title,  # Use adjusted title
                            tickmode='array',
                            tickvals=list(range(len(grouped_df))),
                            ticktext=tick_labels  # Use adjusted tick labels
                        )
                    )

                    # Truncate legend labels
                    fig.for_each_trace(lambda t: t.update(
                        name=t.name[:10] + "..." if len(t.name) > 10 else t.name))

                    st.plotly_chart(fig)

                    # Visualization - Percentage Change
                    fig2 = px.bar(
                        grouped_df, x=group_col, y='Percentage Change',
                        title=f"Percentage Change in {selected_pre_col}",
                        text=grouped_df['Percentage Change'].apply(
                            lambda x: f"{x:.2f}%"),
                        labels={'Percentage Change': 'Percentage Change (%)'},
                        color='Percentage Change', color_continuous_scale='RdYlGn',
                        # Ensure full category names appear in hover
                        hover_data={group_col: True}
                    )

                    # Improve readability
                    fig2.update_layout(
                        template="plotly_white",  # Set background to white
                        legend_title="Survey Type",
                        legend_traceorder="normal",
                        xaxis=dict(
                            title=x_axis_title,
                            tickmode='array',
                            tickvals=list(range(len(grouped_df))),
                            # Truncate long labels
                            ticktext=tick_labels
                        )
                    )

                    # Truncate legend labels
                    fig2.for_each_trace(lambda t: t.update(
                        name=t.name[:10] + "..." if len(t.name) > 10 else t.name))

                    st.plotly_chart(fig2)

                    # # 📊 Boxplot - Distribution of Pre vs Post Responses
                    # fig_box = px.box(
                    #     filtered_df,
                    #     y=[selected_pre_col, selected_post_col],
                    #     title="Distribution of Pre vs Post Responses",
                    #     labels={selected_pre_col: "Pre-Survey",
                    #             selected_post_col: "Post-Survey"},
                    #     # Ensure full category names appear in hover
                    #     hover_data={group_col: True} if group_col else None,
                    #     template="plotly_white"  # Set background to white
                    # )

                    # # Improve readability (consistent with bar charts)
                    # fig_box.update_layout(
                    #     legend_title="Survey Type",
                    #     legend_traceorder="normal",
                    #     xaxis=dict(
                    #         title=x_axis_title,
                    #         tickmode='array',
                    #         tickvals=list(range(len(grouped_df))),
                    #         ticktext=tick_labels
                    #     )
                    # )

                    # # Truncate legend labels for better readability
                    # fig_box.for_each_trace(lambda t: t.update(
                    #     name=t.name[:10] + "..." if len(t.name) > 10 else t.name))

                    # st.plotly_chart(fig_box)

                    # 🔬 Statistical Analysis
                    if len(filtered_df) > 1:
                        st.subheader("📊 Statistical Significance Testing")

                        # Drop NaN values in selected columns
                        filtered_data = filtered_df[[selected_pre_col, selected_post_col, group_col]].dropna(
                        ) if group_col else filtered_df[[selected_pre_col, selected_post_col]].dropna()

                        # Function to compute Cohen's d
                        def cohen_d(x, y):
                            return (np.mean(y) - np.mean(x)) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)

                        # Function to classify effect size based on Cohen’s d
                        def classify_effect_size(d):
                            if abs(d) < 0.2:
                                return "Small"
                            elif abs(d) < 0.5:
                                return "Medium"
                            else:
                                return "Large"

                        # Function to compute Confidence Interval for mean difference
                        def compute_confidence_interval(data_pre, data_post):
                            mean_diff = np.mean(data_post - data_pre)
                            std_err = np.std(
                                data_post - data_pre, ddof=1) / np.sqrt(len(data_pre))
                            conf_interval = t.interval(0.95, df=len(
                                data_pre)-1, loc=mean_diff, scale=std_err)
                            return mean_diff, conf_interval

                        # Run t-test, Cohen's d, and Confidence Interval separately for each group
                        if group_col:
                            results = []
                            for group, data in filtered_data.groupby(group_col):
                                if len(data) > 1 and data[selected_pre_col].nunique() > 1 and data[selected_post_col].nunique() > 1:
                                    # Compute t-test
                                    t_stat, p_value = ttest_rel(
                                        data[selected_pre_col], data[selected_post_col])

                                    # Compute Cohen’s d
                                    effect_size = cohen_d(
                                        data[selected_pre_col], data[selected_post_col])
                                    effect_category = classify_effect_size(
                                        effect_size)

                                    # Compute Confidence Interval
                                    mean_diff, conf_interval = compute_confidence_interval(
                                        data[selected_pre_col], data[selected_post_col])

                                    results.append({
                                        "Group": group,
                                        "T-Statistic": t_stat,
                                        "P-Value": p_value,
                                        "Cohen's d": effect_size,
                                        "Effect Size": effect_category,
                                        "Mean Difference": mean_diff,
                                        "95% CI Lower": conf_interval[0],
                                        "95% CI Upper": conf_interval[1]
                                    })

                            if results:
                                results_df = pd.DataFrame(results)
                                st.write(
                                    "### T-Test, Effect Size, & Confidence Intervals by Group")
                                st.dataframe(results_df)

                                # Highlight significant groups
                                significant_groups = results_df[results_df["P-Value"]
                                                                < 0.05]["Group"].tolist()
                                if significant_groups:
                                    st.success(f"✅ The change is statistically significant in the following groups: {
                                               ', '.join(map(str, significant_groups))}")
                                else:
                                    st.warning(
                                        "⚠️ No groups showed statistically significant changes (p >= 0.05).")
                            else:
                                st.warning(
                                    "⚠️ Statistical tests could not be performed due to insufficient variation in responses within groups.")

                        # Run tests on full dataset if no grouping
                        else:
                            if len(filtered_data) > 1 and filtered_data[selected_pre_col].nunique() > 1 and filtered_data[selected_post_col].nunique() > 1:
                                t_stat, p_value = ttest_rel(
                                    filtered_data[selected_pre_col], filtered_data[selected_post_col])
                                effect_size = cohen_d(
                                    filtered_data[selected_pre_col], filtered_data[selected_post_col])
                                effect_category = classify_effect_size(
                                    effect_size)
                                mean_diff, conf_interval = compute_confidence_interval(
                                    filtered_data[selected_pre_col], filtered_data[selected_post_col])

                                st.write(
                                    f"**Overall T-statistic:** {t_stat:.4f}")
                                st.write(f"**Overall P-value:** {p_value:.4f}")

                                st.write(
                                    f"**Overall Cohen's d:** {effect_size:.4f} ({effect_category} Effect)")
                                st.write(f"**95% Confidence Interval for Mean Difference:** ({
                                         conf_interval[0]:.2f}, {conf_interval[1]:.2f})")

                                if p_value < 0.05:
                                    st.success(
                                        "✅ The overall change is statistically significant (p < 0.05).")
                                else:
                                    st.warning(
                                        "⚠️ The overall change is not statistically significant (p >= 0.05).")
                            else:
                                st.warning(
                                    "⚠️ Statistical tests could not be performed due to insufficient variation in responses.")

                    # 📥 Download Processed Data

                    @st.cache_data
                    def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df(merged_df)

                    st.download_button(
                        label="📥 Download Processed Data",
                        data=csv,
                        file_name="processed_survey_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(
                        "No corresponding post-survey column found for selection.")
