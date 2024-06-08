"""
python main.py
streamlit run main.py
https://raybellwaves-rapids-issue-explorer-main-svnwve.streamlit.app/
"""

# ORG = "rapidsai"
# REPO = "cudf"
ORG = "NVIDIA"
REPO = "spark-rapids"
BOTS = ["dependabot[bot]", "GPUtester", "github-actions[bot]"]
import os  # noqa: E402

try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
except KeyError:
    OPENAI_API_KEY = ""


def chat_response(content):
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def num_tokens_from_string(
    string: str,
    encoding_name: str = "cl100k_base",
) -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def agent_response(agent, content):
    return agent.invoke(content)["output"]


def pull_issues(org: str = ORG, repo: str = REPO) -> None:
    # Currently only open issues
    import os
    import requests
    import json

    from tqdm.auto import tqdm

    output_folder = f"{repo}_issues"
    os.makedirs(output_folder, exist_ok=True)

    headers = {"Authorization": f"token {os.environ['GITHUB_API_TOKEN']}"}

    issues = []
    page = 1
    while True:
        # Open issues and PRs
        issues_url = f"https://api.github.com/repos/{org}/{repo}/issues?state=open&per_page=100&page={page}"
        response = requests.get(issues_url, headers=headers)
        if response.status_code != 200:
            break
        page_issues = response.json()
        if not page_issues:
            break
        only_issues = [issue for issue in page_issues if "pull_request" not in issue]
        no_bots = [issue for issue in only_issues if issue["user"]["login"] not in BOTS]
        issues.extend(no_bots)
        page += 1

    for issue in tqdm(issues, "fetching issues"):
        issue_number = issue["number"]
        padded_issue_number = f"{issue_number:05d}"

        # Fetch issue details
        issue_detail_url = (
            f"https://api.github.com/repos/{org}/{repo}/issues/{issue_number}"
        )
        issue_detail_response = requests.get(issue_detail_url, headers=headers)
        issue_detail = issue_detail_response.json()

        # Save issue details to a file
        file_path = os.path.join(
            output_folder, f"issue_detail_{padded_issue_number}.json"
        )
        with open(file_path, "w") as f:
            json.dump(issue_detail, f, indent=4)

    return None


def concat_issues(repo: str = REPO) -> None:
    import json
    import os
    import pandas as pd

    from tqdm.auto import tqdm

    df = pd.DataFrame()
    for file in tqdm(sorted(os.listdir(f"{repo}_issues")), "concatenating issues"):
        with open(f"{repo}_issues/{file}", "r") as f:
            data = json.load(f)
        _df = pd.json_normalize(data)
        if _df["body"][0] is None:
            _df["body"] = ""

        _df["label_names"] = _df["labels"].apply(
            lambda x: [label["name"] for label in x] if isinstance(x, list) else []
        )
        # Use an LLM to help categorize issues
        _df["LLM_title_subject"] = chat_response(
            f"Give me a one word summary of the following GitHub {repo} issue title: {_df['title'][0]}"
        )
        # Count tokens in issue
        try:
            _df["issue_text_tokens"] = num_tokens_from_string(
                _df["body"][0], "cl100k_base"
            )
        except TypeError:
            _df["issue_text_tokens"] = -1

        # TODO remove issue template?
        df = pd.concat([df, _df], axis=0).reset_index(drop=True)

    df = df.rename(
        columns={
            "comments": "n_comments",
            "user.login": "issue_user.login",
            "body": "issue_text",
            "reactions.total_count": "issue_reactions.total_count",
            "reactions.+1": "issue_reactions.+1",
            "reactions.-1": "issue_reactions.-1",
            "reactions.laugh": "issue_reactions.laugh",
            "reactions.hooray": "issue_reactions.hooray",
            "reactions.confused": "issue_reactions.confused",
            "reactions.heart": "issue_reactions.heart",
            "reactions.rocket": "issue_reactions.rocket",
            "reactions.eyes": "issue_reactions.eyes",
            "created_at": "issue_created_at",
            "updated_at": "issue_updated_at",
        }
    )

    df.to_csv(f"{repo}_issue_details.csv")
    # Unique issue creators
    df.rename(
        columns={
            "issue_user.login": "user.login",
        }
    )["user.login"].drop_duplicates().reset_index(drop=True).to_csv(
        f"{repo}_issue_posters.csv"
    )

    return None


def pull_comments(repo: str = REPO) -> None:
    # Pull comments for issues already pulled
    import os
    import requests
    import pandas as pd
    import json

    from tqdm.auto import tqdm

    output_folder = f"{repo}_comments"
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(f"{repo}_issue_details.csv")

    headers = {"Authorization": f"token {os.environ['GITHUB_API_TOKEN']}"}

    for url in tqdm(df["comments_url"], "fetching comments"):
        issue = url.split("/")[-2]
        padded_issue = f"{int(issue):05d}"
        comment_detail_response = requests.get(url, headers=headers)
        comment_detail = comment_detail_response.json()
        file_path = os.path.join(output_folder, f"issue_comment_{padded_issue}.json")
        with open(file_path, "w") as f:
            json.dump(comment_detail, f, indent=4)

    return None


def concat_comments(repo: str = REPO) -> None:
    # One row is one comment
    import json
    import os
    import pandas as pd

    from tqdm.auto import tqdm

    df = pd.DataFrame()
    for file in tqdm(sorted(os.listdir(f"{repo}_comments")), "concatenating comments"):
        with open(f"{repo}_comments/{file}", "r") as f:
            data = json.load(f)
        _df = pd.json_normalize(data)
        df = pd.concat([df, _df], axis=0).reset_index(drop=True)

    # Add number to join with issues
    df["number"] = (
        df["html_url"].str.split("/", expand=True)[6].str.split("#", expand=True)[0]
    ).astype(int)

    df = df.rename(
        columns={
            "comments": "n_comments",
            "user.login": "comment_user.login",
            "body": "comment_text",
            "reactions.total_count": "comment_reactions.total_count",
            "reactions.+1": "comment_reactions.+1",
            "reactions.-1": "comment_reactions.-1",
            "reactions.laugh": "comment_reactions.laugh",
            "reactions.hooray": "comment_reactions.hooray",
            "reactions.confused": "comment_reactions.confused",
            "reactions.heart": "comment_reactions.heart",
            "reactions.rocket": "comment_reactions.rocket",
            "reactions.eyes": "comment_reactions.eyes",
            "created_at": "comment_created_at",
            "updated_at": "comment_updated_at",
        }
    )

    df.to_csv(f"{repo}_comment_details.csv")
    # Unique commenters
    df.rename(
        columns={
            "comment_user.login": "user.login",
        }
    )["user.login"].drop_duplicates().reset_index(drop=True).to_csv(
        f"{repo}_comment_commenters.csv"
    )
    return None


def pull_users(repo: str = REPO) -> None:
    import os
    import requests
    import pandas as pd
    import json

    from tqdm.auto import tqdm

    output_folder = f"{repo}_users"
    os.makedirs(output_folder, exist_ok=True)

    df_posters = pd.read_csv(f"{repo}_issue_posters.csv")
    df_commenters = pd.read_csv(f"{repo}_comment_commenters.csv")
    df = pd.concat([df_posters, df_commenters], axis=0)[
        ["user.login"]
    ].drop_duplicates()

    headers = {"Authorization": f"token {os.environ['GITHUB_API_TOKEN']}"}

    for username in tqdm(df["user.login"], "fetching data for users"):
        user_detail_response = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers,
        )
        user_detail = user_detail_response.json()
        file_path = os.path.join(output_folder, f"user_detail_{username}.json")
        with open(file_path, "w") as f:
            json.dump(user_detail, f, indent=4)
    return None


def concat_users(repo: str = REPO) -> None:
    import json
    import os
    import numpy as np
    import pandas as pd
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderUnavailable

    from tqdm.auto import tqdm

    geolocator = Nominatim(user_agent="_", timeout=10)

    df = pd.DataFrame()
    for file in tqdm(sorted(os.listdir(f"{repo}_users")), "concatenating users"):
        with open(f"{repo}_users/{file}", "r") as f:
            data = json.load(f)
        _df = pd.json_normalize(data)
        _df["name_company"] = f"{_df['name'].values[0]} ({_df['company'].values[0]})"
        try:
            geocoded_location = geolocator.geocode(_df["location"].values)
            _df["location_lat"] = geocoded_location.latitude
            _df["location_lon"] = geocoded_location.longitude
        except GeocoderUnavailable:
            pass
        except AttributeError:
            _df["location_lat"] = np.nan
            _df["location_lon"] = np.nan

        df = pd.concat([df, _df], axis=0).reset_index(drop=True)
    df = df.rename(columns={"login": "user.login"})
    df.to_csv(f"{repo}_user_details.csv")
    return None


def create_dataframe(repo: str = REPO) -> None:
    # Create a single dataframe

    # Filter out bots

    import pandas as pd

    issue_core_columns = [
        "number",
        "title",
        "issue_text",
        "LLM_title_subject",
        "issue_text_tokens",
        "issue_user.login",
        "author_association",
        "label_names",
        # "state",
        # "locked",
        # "milestone",
        "issue_created_at",
        "issue_updated_at",
        "issue_reactions.total_count",
        "issue_reactions.+1",
        "issue_reactions.-1",
        "issue_reactions.laugh",
        "issue_reactions.hooray",
        "issue_reactions.confused",
        "issue_reactions.heart",
        "issue_reactions.rocket",
        "issue_reactions.eyes",
        "n_comments",
    ]
    df_issues = pd.read_csv(f"{repo}_issue_details.csv")[issue_core_columns]
    df_issues = df_issues.loc[~df_issues["issue_user.login"].isin(BOTS)].reset_index(
        drop=True
    )

    comment_core_columns = [
        "number",
        "comment_text",
        "comment_user.login",
        "comment_created_at",
        "comment_updated_at",
        "comment_reactions.total_count",
        "comment_reactions.+1",
        "comment_reactions.-1",
        "comment_reactions.laugh",
        "comment_reactions.hooray",
        "comment_reactions.confused",
        "comment_reactions.heart",
        "comment_reactions.rocket",
        "comment_reactions.eyes",
    ]
    df_comments = pd.read_csv(f"{repo}_comment_details.csv")[comment_core_columns]
    df_comments = df_comments.loc[
        ~df_comments["comment_user.login"].isin(BOTS)
    ].reset_index(drop=True)

    df = df_issues.merge(df_comments, how="outer")

    user_core_columns = [
        "user.login",
        "name",
        "email",
        "company",
        "name_company",
        "location",
        "location_lat",
        "location_lon",
        "followers",
    ]
    df_users = pd.read_csv(f"{repo}_user_details.csv")[user_core_columns]
    df_users = df_users.loc[~df_users["user.login"].isin(BOTS)].reset_index(drop=True)

    df = df.merge(df_users, left_on="issue_user.login", right_on="user.login").rename(
        columns={
            "email": "issue_user.login_email",
            "name": "issue_user.login_name",
            "company": "issue_user.login_company",
            "name_company": "issue_user.login_name_company",
            "location": "issue_user.login_location",
            "location_lat": "issue_user.login_location_lat",
            "location_lon": "issue_user.login_location_lon",
            "followers": "issue_user.login_followers",
        }
    )
    df = df.merge(
        df_users, how="left", left_on="comment_user.login", right_on="user.login"
    ).rename(
        columns={
            "email": "comment_user.login_email",
            "name": "comment_user.login_name",
            "company": "comment_user.login_company",
            "name_company": "comment_user.login_name_company",
            "location": "comment_user.login_location",
            "location_lat": "comment_user.login_location_lat",
            "location_lon": "comment_user.login_location_lon",
            "followers": "comment_user.login_followers",
        }
    )
    order_cols = [
        "number",
        "title",
        "issue_text",
        "LLM_title_subject",
        "issue_text_tokens",
        "label_names",
        "issue_user.login",
        "author_association",
        "issue_user.login_name",
        "issue_user.login_company",
        "issue_user.login_name_company",
        "issue_user.login_email",
        "issue_user.login_followers",
        "issue_user.login_location",
        "issue_user.login_location_lat",
        "issue_user.login_location_lon",
        "issue_created_at",
        "issue_updated_at",
        "issue_reactions.total_count",
        "issue_reactions.+1",
        "issue_reactions.-1",
        "issue_reactions.laugh",
        "issue_reactions.hooray",
        "issue_reactions.confused",
        "issue_reactions.heart",
        "issue_reactions.rocket",
        "issue_reactions.eyes",
        "n_comments",
        "comment_text",
        "comment_user.login",
        "comment_user.login_name",
        "comment_user.login_company",
        "comment_user.login_name_company",
        "comment_user.login_email",
        "comment_user.login_followers",
        "comment_user.login_location",
        "comment_user.login_location_lat",
        "comment_user.login_location_lon",
        "comment_created_at",
        "comment_updated_at",
        "comment_reactions.total_count",
        "comment_reactions.+1",
        "comment_reactions.-1",
        "comment_reactions.laugh",
        "comment_reactions.hooray",
        "comment_reactions.confused",
        "comment_reactions.heart",
        "comment_reactions.rocket",
        "comment_reactions.eyes",
    ]
    df = df[order_cols]

    df.to_parquet(f"{repo}_issue_with_comments.parquet")

    # Small version with just issue and some stats from comments
    issue_user_cols = [
        "issue_user.login_email",
        "issue_user.login_name",
        "issue_user.login_company",
        "issue_user.login_name_company",
        "issue_user.login_location",
        "issue_user.login_location_lat",
        "issue_user.login_location_lon",
        "issue_user.login_followers",
    ]
    df_issue_summary = df[issue_core_columns + issue_user_cols].drop_duplicates()
    commenters = (
        df.groupby("number")["comment_user.login_name_company"].agg(list).reset_index()
    ).rename(columns={"comment_user.login_name_company": "commenters"})
    comment_reactions = (
        df.groupby("number")["comment_reactions.total_count"].sum().reset_index()
    ).astype(int)
    df_issue_summary = df_issue_summary.merge(commenters).merge(comment_reactions)
    df_issue_summary.to_parquet(f"{repo}_issue_summary.parquet")
    return None


def create_vector_db(repo: str = REPO) -> None:
    import pickle
    import pandas as pd
    from langchain_openai import OpenAIEmbeddings
    from pymilvus import MilvusClient

    df = pd.read_parquet(f"{repo}_issue_summary.parquet")
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    embeddings = embeddings_model.embed_documents(
        df["issue_text"].fillna("").values
    )  # ndocs x 1536
    with open(f"{repo}_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    data = [
        {
            "id": row["number"],
            "vector": embeddings[i],
            "text": row["issue_text"],
            "subject": row["LLM_title_subject"],
        }
        for i, row in df.iterrows()
    ]

    client = MilvusClient(f"./milvus_{repo.replace('-', '_')}.db")
    client.create_collection(
        collection_name=f"{repo.replace('-', '_')}_issue_text", dimension=1536
    )
    _ = client.insert(collection_name=f"{repo.replace('-', '_')}_issue_text", data=data)


def query_data(repo: str = REPO) -> None:
    import pandas as pd
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import OpenAI as OpenAI_langchain
    from langchain_openai import OpenAIEmbeddings
    from pymilvus import MilvusClient

    df_issues = pd.read_parquet(f"{repo}_issue_summary.parquet")
    # drop the issue_text column as context is too large for agent
    # we will use the vector database instead
    df_issues = df_issues.drop(
        columns=[
            "issue_text",
            "LLM_title_subject",
            "label_names",
            "issue_text_tokens",
            "issue_created_at",
            "issue_updated_at",
            "issue_reactions.+1",
            "issue_reactions.-1",
            "issue_reactions.laugh",
            "issue_reactions.hooray",
            "issue_reactions.confused",
            "issue_reactions.heart",
            "issue_reactions.rocket",
            "issue_reactions.eyes",
            "issue_user.login_location_lat",
            "issue_user.login_location_lon",
        ]
    )
    # Create simple column names to help the agent
    df_issues = df_issues.rename(
        columns={
            "number": f"{repo}_issue_number",
            "title": f"{repo}_issue_title",
            "author_association": f"association_to_{repo}",
            "issue_reactions.total_count": "number_of_reactions_on_issue",
            "n_comments": "number_of_comments",
            "issue_user.login_email": "email",
            "issue_user.login_name": "name",
            "issue_user.login_company": "company",
            "issue_user.login_name_company": "name_company",
            "issue_user.login_location": "location",
            "issue_user.login_followers": "github_followers",
            "comment_reactions.total_count": "number_of_reactions_on_comments",
        }
    )
    print(df_issues["company"].value_counts())
    agent = create_pandas_dataframe_agent(
        OpenAI_langchain(
            temperature=0,
            model="gpt-3.5-turbo-instruct",
            openai_api_key=OPENAI_API_KEY,
        ),
        df_issues,
        verbose=True,
    )
    if repo == "cudf":
        _company = "Walmart"
    elif repo == "spark-rapids":
        _company = "bytedance"
    print(df_issues[df_issues["company"] == _company])
    response = agent_response(agent, f"What issues have {_company} created?")
    print(response)
    if ":" in response:
        response = response.split(":")[1].strip()

    question = f"What issues are similar to {response}?"
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    question_embeddings = embeddings_model.embed_documents([question])
    client = MilvusClient(f"./milvus_{repo.replace('-', '_')}.db")
    res = client.search(
        collection_name=f"{repo.replace('-', '_')}_issue_text",
        data=question_embeddings,
        limit=2,
    )
    similar_issue_1 = res[0][0]["id"]
    similar_issue_2 = res[0][1]["id"]
    print(df_issues[df_issues[f"{repo}_issue_number"] == similar_issue_1])
    print(df_issues[df_issues[f"{repo}_issue_number"] == similar_issue_2])


def steamlit_dashboard():
    from openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    from streamlit_folium import st_folium
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import OpenAI as OpenAI_langchain
    from pymilvus import MilvusClient
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    def chat_response(content):
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content

    def agent_response(agent, content):
        return agent.invoke(content)["output"]

    @st.cache_data
    def fetch_issue_data(repo: str):
        df_issues = pd.read_parquet(f"{repo}_issue_summary.parquet")
        # Create simple column names to help the agent
        df_issues = df_issues.rename(
            columns={
                "number": f"{repo}_issue_number",
                "title": f"{repo}_issue_title",
                "author_association": f"association_to_{repo}",
                "issue_reactions.total_count": "number_of_reactions_on_issue",
                "n_comments": "number_of_comments",
                "issue_user.login": "github_username",
                "issue_user.login_email": "email",
                "issue_user.login_name": "name",
                "issue_user.login_company": "company",
                "issue_user.login_name_company": "name_company",
                "issue_user.login_location": "location",
                "issue_user.login_location_lat": "location_lat",
                "issue_user.login_location_lon": "location_lon",
                "issue_user.login_followers": "github_number_of_followers",
                "comment_reactions.total_count": "number_of_reactions_on_comments",
            }
        )
        return df_issues

    repo = st.selectbox("Select a repository", ["cudf", "spark-rapids"])

    st.title(f"{repo} GitHub issue explorer for DevRel")

    st.markdown(
        """
    This dashboard can help with:
    - Identification of users/leads
    - Identify common developer pain points
    - Identify most common requested features
    """
    )

    st.markdown(
        "**This dashboard is WIP and may be prone to errors.** "
        "Code associated with this can be found at "
        "https://github.com/raybellwaves/rapids-issue-explorer"
    )

    st.subheader("Partners")

    st.markdown(
        f"We can explore what companies are active in in the {repo} GitHub repo. "
        "These are likely super users:"
    )
    df_issues = fetch_issue_data(repo=repo)
    company_counts = df_issues["company"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    company_counts.plot(
        kind="bar",
        ax=ax,
        xlabel="Company",
        ylabel="Count",
        title=f"Count of companies who post and comment on the {repo} GitHub repo",
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        "To learn more about these companies we will ask a LLM questions such as "
        "**'What type of company is Halliburton?'** "
        "or generic questions such as "
        "**'Why would Halliburton use GPUs to speed up data processing?'** "
        "but don't expect a good result. "
        "We will use the GitHub data to refine this question later. "
    )

    st.markdown("**You will need to pass an OpenAI API key to ask questions below:**")
    openai_api_key = st.text_input("OpenAI API Key:", type="password")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    if repo == "cudf":
        _company = "Walmart"
    elif repo == "spark-rapids":
        _company = "bytedance"
    content = st.text_input(
        f"Ask questions about companies who use {repo}:",
        f"What type of company is {_company}?",
    )
    if openai_api_key:
        st.write(chat_response(content))

    st.subheader("Community building")

    st.markdown(
        f"We can explore the location of users who post on {repo}. "
        "This can help with event planning and community building. "
    )

    _df = df_issues[
        [
            "github_username",
            "name",
            "company",
            f"association_to_{repo}",
            "location",
            "github_number_of_followers",
            "location_lat",
            "location_lon",
        ]
    ].drop_duplicates()
    gdf = gpd.GeoDataFrame(
        _df,
        geometry=gpd.points_from_xy(
            _df["location_lon"],
            _df["location_lat"],
        ),
        crs="epsg:4326",
    )
    m = gdf.explore(
        column=f"association_to_{repo}",
        cmap="viridis",
        legend=True,
    )
    st_folium(m, width=1000)

    st.subheader("Understanding developers")

    st.markdown(
        f"""
        We can explore the GitHub data to understand what developers are interested in 
        and to ensure their requested features or bug are taken into account in the roadmap
        You can ask questions such as: 
        - **What issues are {_company} most interested in?**
        - **What issue is ETH Zürich most interested in?**
        - **What issue has the most reactions?**
        - **What company posted the issue with the most reactions?**
        - **What are the top 5 issues with the most most reactions?**
        """
    )
    df_issues = df_issues.drop(
        columns=[
            "issue_text",
            "LLM_title_subject",
            "label_names",
            "issue_text_tokens",
            "issue_created_at",
            "issue_updated_at",
            "issue_reactions.+1",
            "issue_reactions.-1",
            "issue_reactions.laugh",
            "issue_reactions.hooray",
            "issue_reactions.confused",
            "issue_reactions.heart",
            "issue_reactions.rocket",
            "issue_reactions.eyes",
            "location_lat",
            "location_lon",
        ]
    )
    if openai_api_key:
        agent = create_pandas_dataframe_agent(
            OpenAI_langchain(
                temperature=0,
                model="gpt-3.5-turbo-instruct",
                openai_api_key=openai_api_key,
            ),
            df_issues,
            verbose=True,
        )
    content = st.text_input(
        f"Ask questions about about external {repo} users and developers using the GitHub data:",
        f"What issues has the company {_company} created?",
    )
    if openai_api_key:
        response = agent_response(agent, content)
        st.write(response)
        if ":" in response:
            response = response.split(":")[1].strip()

    st.markdown("We will now use a vector database to query matching issues:")
    if openai_api_key:
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
        question = f"What issues are similar to {response}?"
        question_embeddings = embeddings_model.embed_documents([question])
        milvus_client = MilvusClient(f"./milvus_{repo.replace('-', '_')}.db")
        res = milvus_client.search(
            collection_name=f"{repo.replace('-', '_')}_issue_text",
            data=question_embeddings,
            limit=3,
        )
        similar_issue_1 = res[0][0]["id"]
        similar_issue_2 = res[0][1]["id"]
        similar_issue_3 = res[0][1]["id"]
        st.dataframe(df_issues[df_issues[f"{repo}_issue_number"] == similar_issue_1])
        st.dataframe(df_issues[df_issues[f"{repo}_issue_number"] == similar_issue_2])
        if similar_issue_3 != similar_issue_2:
            st.dataframe(
                df_issues[df_issues[f"{repo}_issue_number"] == similar_issue_3]
            )


if __name__ == "__main__":
    """
    python main.py
    streamlit run main.py
    https://raybellwaves-rapids-issue-explorer-main-svnwve.streamlit.app/
    """
    # pull_issues()
    # concat_issues()
    # pull_comments()
    # concat_comments()
    # pull_users()
    # concat_users()
    # create_dataframe()
    # create_vector_db(repo="cudf")
    # query_data("cudf")
    steamlit_dashboard()
