import streamlit as st
import pandas as pd
import re
import json
from collections import defaultdict
import openai
import sys
import difflib
import Levenshtein

# Set your OpenAI API key

def load_openai_key(path="openai_key.txt"):
    try:
        with open(path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        sys.exit("OpenAI API key file not found. Please create a file named `.openai_key` with your API key.")

openai.api_key = load_openai_key()

def is_similar(str1, str2, threshold=0.85):
    """Check if two strings are similar based on Levenshtein ratio."""
    return Levenshtein.ratio(str1, str2) > threshold

def remove_similar_entries(rule_descriptions, threshold=0.85):
    """Remove similar rule descriptions, keeping only one representative."""
    unique_rules = []
    for rule in rule_descriptions:
        if not any(is_similar(rule, existing_rule, threshold) for existing_rule in unique_rules):
            unique_rules.append(rule)
    # print(rule_descriptions[3])
    # print(rule_descriptions[4])
    # print(Levenshtein.ratio(rule_descriptions[3], rule_descriptions[4]))
    return unique_rules

def extract_cuda_kernels(content):
    """Extract CUDA kernel signatures from the given code content."""
    kernel_pattern = r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)"
    matches = re.findall(kernel_pattern, content)
    return [f"{name}({params})" for name, params in matches]

def find_large_stalls(data, threshold):
    """Find line numbers where stall ratios exceed the threshold and aggregate duplicate entries."""
    aggregated_stalls = defaultdict(set)
    for _, row in data.iterrows():
        line_no = row.get("Line No", None)
        high_stalls = [col for col in data.columns if 'stall' in col.lower() and 'ratio' in col.lower() and row[col] > threshold]
        if high_stalls and line_no is not None:
            aggregated_stalls[line_no].update(high_stalls)
    return [{"Line No": line_no, "Stall Reasons": list(stalls)} for line_no, stalls in aggregated_stalls.items()]

# def submit_query_to_chatgpt(content):
#     """Submit a query to ChatGPT with the uploaded CUDA code."""
#     query = f"Given the code:\n\n{content}\n\nthat ran on an NVIDIA H100 machine, can you optimize this code for better runtime? Can you give me only the optimized code without any explanation."
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an AI coding assistant."},
#                 {"role": "user", "content": query}
#             ],
#             max_tokens=5000,
#             temperature=0.7
#         )
#         return response.choices[0].message['content'].strip()
#     except Exception as e:
#         return f"An error occurred while contacting ChatGPT: {e}"

# def submit_query_to_chatgpt_new(content):
#     query = content
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are an AI coding assistant."},
#                 {"role": "user", "content": query}
#             ],
#             max_tokens=5000,
#             temperature=0.7
#         )
#         return response.choices[0].message['content'].strip()
#     except Exception as e:
#         return f"An error occurred while contacting ChatGPT: {e}"  

def submit_query(content):
    query = content
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI coding assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=5000,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred while contacting ChatGPT: {e}"  
    
# def prepare_sources_to_submit(content):
#     query = f"Given the code:\n\n{content}\n\nthat ran on an NVIDIA H100 machine, can you optimize this code for better runtime? Do not add any explanantion, just give me the optimized code only."
#     # submit_query(content=query)
#     return query

# def prepare_sources_to_submit(content, pc_data):
#     query_with_sampling = f"Given the code:\n\n{content}\n\nand the following PC sampling data insights:\n"
#     large_stalls = find_large_stalls(pc_data, threshold=0.3)
#     for stall in large_stalls:
#         query_with_sampling += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
#     query_with_sampling += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided for better runtime? Do not add any explanantion, just give me the optimized code only."
#     return query_with_sampling


# def prepare_sources_to_submit(content, important_counters):
#     query_with_json = f"Given the code:\n\n{content}\n\nand the following data where the values represent the affect of that particular hardware counter on runtime.\n"
#     for counter in important_counters:
#         query_with_json += f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, Value: {counter['Value']}\n"
#     query_with_json += "\nThis code ran on an NVIDIA A100 machine. Can you optimize this code based on the insights provided for better runtime?? Do not add any explanantion, just give me the optimized code only."
#     return query_with_json

                
# def prepare_sources_to_submit(content, filtered_roofline_data):                
#     roofline_query = f"Given the code:\n\n{content}\n\nand the following Roofline analysis data:\n{filtered_roofline_data}\nCan you optimize this code for better runtime? Do not add any explanantion, just give me the optimized code only."
#     return roofline_query

# def prepare_sources_to_submit(content,pc_data, important_counters):
#     combined_query = f"Given the code:\n\n{content}\n\nand the following PC sampling data insights:\n"
#     large_stalls = find_large_stalls(pc_data, threshold=0.3)
#     for stall in large_stalls:
#         combined_query += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
#     combined_query += "\nAdditionally, the following counters from the data where the values represent the affect of that particular hardware counter on runtime.\n"
#     for counter in important_counters:
#         combined_query += f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, Value: {counter['Value']}\n"
#     combined_query += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided for better runtime? Do not add any explanantion just give me the optimized code only?"
#     return combined_query 

# def prepare_sources_to_submit(content,pc_data, filtered_roofline_data):
#     combined_query = f"Given the code:\n\n{content}\n\nand the following PC sampling data insights:\n"
#     large_stalls = find_large_stalls(pc_data, threshold=0.3)
#     for stall in large_stalls:
#         combined_query += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
#     combined_query += "\nAdditionally, the following Roofline analysis data was collected:\n{filtered_roofline_data}\n"
#     combined_query += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided for better runtime? Do not add any explanantion just give me the optimized code only?"
#     return combined_query

# def prepare_sources_to_submit(content, filtered_roofline_data, important_counters):
#     combined_query = f"Given the code:\n\n{content}\n\nand the following PC sampling data insights:\n"
#     combined_query += "\nAdditionally, the following counters from the data where the values represent the affect of that particular hardware counter on runtime.\n"
#     for counter in important_counters:
#         combined_query += f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, Value: {counter['Value']}\n"
#     combined_query += "\nAdditionally, the following Roofline analysis data was collected:\n{filtered_roofline_data}\n"
#     combined_query += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided for better runtime? Do not add any explanantion just give me the optimized code only?"
#     return combined_query 

# def prepare_sources_to_submit(content,pc_data, important_counters, filtered_roofline_data):
#     combined_query = f"Given the code:\n\n{content}\n\nand the following PC sampling data insights:\n"
#     large_stalls = find_large_stalls(pc_data, threshold=0.3)
#     for stall in large_stalls:
#         combined_query += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
#     combined_query += "\nAdditionally, the following counters from the data where the values represent the affect of that particular hardware counter on runtime.\n"
#     for counter in important_counters:
#         combined_query += f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, Value: {counter['Value']}\n"
#     combined_query += "\nAdditionally, the following Roofline analysis data was collected:\n{filtered_roofline_data}\n"
#     combined_query += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided for better runtime? Do not add any explanantion just give me the optimized code only?"
#     return combined_query

# def prepare_sources_to_submit(content, pc_data=None, important_counters=None, filtered_roofline_data=None):
#     combined_query = f"Given the code:\n\n{content}\n"
#     if pc_data is not None:
#         combined_query += "\nand the following PC sampling data insights:\n"
#         large_stalls = find_large_stalls(pc_data, threshold=0.1)
#         for stall in large_stalls:
#             combined_query += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
#     if important_counters is not None:
#         combined_query += "\nI have also found the main hardware performance events that cause this code to use more runtime are\n"
#         for counter in important_counters:
#             combined_query += f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, Value: {counter['Value']}, Description: {counter['Description']}\n"
#         combined_query+="The higher the attached value is the more effect it has on runtime.\n"
#     if filtered_roofline_data is not None:
#         for key, value in filtered_roofline_data.items():
#             comms = ''
#             for x in range(len(value)):
#                 comms += str(x+1) + '. ' + value[x] + "\n"
#             combined_query += f"\nAdditionally, the Roofline comment for {key} kernel were : {comms}\n"
#         # combined_query += f"\nAdditionally, the following Roofline analysis data was collected:{filtered_roofline_data}\n"
#         # for index, row in filtered_roofline_data.iterrows():
#         #     combined_query += ''
#     combined_query += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided for better runtime and give me a complete code to replace?\n"
#     combined_query += "\nAfter the optimized code, explain why you suggested each optimization in terms of the data that I provided. Be concise, to the point, matter of fact, and substantiate every decision using data or references."
#     return combined_query


def prepare_sources_to_submit(content, pc_data=None, important_counters=None, filtered_roofline_data=None):
    combined_query = f"***Code:***\n\n{content}\n"
    if pc_data is not None:
        combined_query += "\nSTALL ANALYSIS:\n"
        large_stalls = find_large_stalls(pc_data, threshold=0.1)
        for stall in large_stalls:
            combined_query += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
    if important_counters is not None:
        new_query = 'can you expand on the following'
        # new_query += "\nROOFLINE ANALYSIS\n"
        for counter in important_counters:
            new_query += f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, Value: {counter['Value']}, Description: {counter['Description']}\n"
        new_query+="into a block that reads as a performance expert’s optinion of what is wrong with the performance of this code so that it can inform an LLM like yourself how to optimize the code’s performance? Be mindful of the number of tokens, we want to be succinct so that we can use the least amount of tokens to pack a punch."
        combined_query += submit_query(new_query)
    if filtered_roofline_data is not None:
        for key, value in filtered_roofline_data.items():
            comms = ''
            for x in range(len(value)):
                comms += str(x+1) + '. ' + value[x] + "\n"
            combined_query += f"\nROOFLINE ANALYSIS for {key} kernel were : {comms}\n"
        # combined_query += f"\nAdditionally, the following Roofline analysis data was collected:{filtered_roofline_data}\n"
        # for index, row in filtered_roofline_data.iterrows():
        #     combined_query += ''
    # combined_query += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided for better runtime and give me a complete code to replace?\n"
    # combined_query += "\nAfter the optimized code, explain why you suggested each optimization in terms of the data that I provided. Be concise, to the point, matter of fact, and substantiate every decision using data or references."
    combined_query += "\nPlease optimize the code based on the above analysis, and explain your changes.\n"
    return combined_query


def extract_important_counters(json_data, threshold=0.1):
    """Extract important counters and their group names based on their values."""
    with open("gpu_counters_description.json") as f:
        counter_descriptions = json.load(f)

    # print(counter_descriptions["sm__inst_executed.avg.per_cycle_active"])
    important_counters = []
    for kernel, groups in json_data.items():
        for group_name, counters in groups.items():
            for counter_name, value in counters.items():
                if value > threshold:
                    important_counters.append({
                        "Kernel": kernel,
                        "Group Name": group_name,
                        "Counter Name": counter_name,
                        "Value": value,
                        "Description": counter_descriptions[counter_name] if counter_name in counter_descriptions else "None"
                    })
    return important_counters



def main():
    st.set_page_config(layout="wide")
    st.title("Code and PC Sampling Analyzer")
    st.write("Upload your code file, PC sampling data file, and optional JSON file for analysis.")
    col1,col2,col3 = st.columns(3)

    # Code file uploader
    uploaded_file = col1.file_uploader("Choose a code file", type=["py", "java", "cpp", "txt", "cu", "cuh"], key="code_file")

    query = None
    chatgpt_response = None
    pc_data = None
    json_data = None
    roofline_data = None

    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")

            # Extract and display CUDA kernel signatures
            if uploaded_file.name.endswith(('.cu', '.cuh')):
                col1.subheader("CUDA Kernel Signatures")
                kernels = extract_cuda_kernels(content)
                if kernels:
                    col1.write("Found the following kernel signatures:")
                    for kernel in kernels:
                        col1.write(f"- {kernel}")
                else:
                    col1.write("No CUDA kernels found in the file.")

            # Button to submit only the code
            if col2.button("Submit Code Only"):
                # chatgpt_response = submit_query_to_chatgpt(content)
                query = prepare_sources_to_submit(content=content)
                chatgpt_response = submit_query(query)

        except UnicodeDecodeError:
            col1.error("Unable to decode the file content. Please make sure it is a text-based code file.")

    # PC sampling data uploader
    pc_sampling_file = col1.file_uploader("Optionally upload a file describing PC sampling data", type=["csv", "txt"], key="pc_sampling_file")

    if pc_sampling_file is not None:
        try:
            pc_data = pd.read_csv(pc_sampling_file)
            with col1.expander("PC Sampling Data Insights"):
                col1.write("PC Sampling Data Loaded Successfully.")
                large_stalls = find_large_stalls(pc_data, threshold=0.05)
                if large_stalls:
                    col1.write("Lines with High Stall Ratios:")
                    for stall in large_stalls:
                        col1.write(f"Line {stall['Line No']}: {', '.join(stall['Stall Reasons'])}")
                else:
                    col1.write("No lines found with stall ratios exceeding 30%.")

            # Button to submit code with PC sampling data
            if uploaded_file is not None and col2.button("Submit Code with PC Sampling Data"):
                # query_with_sampling = f"Given the code:\n\n{content}\n\nand the following PC sampling data insights:\n"
                # for stall in find_large_stalls(pc_data, threshold=0.3):
                #     query_with_sampling += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
                # query_with_sampling += "\nThis code ran on an NVIDIA H100 machine. Can you optimize this code based on the insights provided? Can you give me only the optimized code without any explanation."
                # col3.subheader("ChatGPT Query")
                # col3.text_area("Query: ", query_with_json, height=300)
                query = prepare_sources_to_submit(content=content,pc_data=pc_data)
                chatgpt_response = submit_query(query)

        except Exception as e:
            col1.error(f"Error processing PC sampling data: {e}")

    # JSON file uploader
    json_file = col1.file_uploader("Optionally upload the importance results for analysis", type=["json"], key="json_file")

    if json_file is not None:
        try:
            json_data = json.load(json_file)
            important_counters = extract_important_counters(json_data, threshold=0.1)
            with col1.expander("Important Counters and Groups"):
                if important_counters:
                    col1.write("Extracted Important Counters:")
                    col1.write(pd.DataFrame(important_counters))
                else:
                    col1.write("No counters exceeded the importance threshold.")

            # Button to submit code with JSON insights
            if uploaded_file is not None and col2.button("Submit Code with importance Analysis"):
                # roofline_query = f"Given the code:\n\n{content}\n\nand the following Roofline analysis data:\n{filtered_roofline_data}\nCan you optimize this code? Give me the optimized code only."
                # chatgpt_response = submit_query_to_chatgpt(roofline_query)
                query = prepare_sources_to_submit(content=content, important_counters=important_counters) 
                chatgpt_response = submit_query(query)

        except Exception as e:
            col1.error(f"Error processing importance analysis file: {e}")\
            
    # Roofline data uploader
    roofline_file = col1.file_uploader("Optionally upload a Roofline Data CSV file", type=["csv"], key="roofline_file")

    if roofline_file is not None:
        try:
            roofline_data = pd.read_csv(roofline_file)
            nadropped_roofline_data = roofline_data.dropna(subset=["Rule Name", "Rule Description", "Kernel Name"])
            filtered_roofline_data = nadropped_roofline_data[["Rule Name", "Rule Description", "Kernel Name"]]
            # with col1.expander("Roofline Data Insights"):
            #     col1.write("Filtered Roofline Data:")
            #     col1.write(filtered_roofline_data.head())

            required_columns = {"Kernel Name", "Rule Description", "Estimated Speedup"}
            if not required_columns.issubset(nadropped_roofline_data.columns):
                raise ValueError(f"Missing required columns in CSV file: {required_columns - set(nadropped_roofline_data.columns)}")

            # Convert "Estimated Speedup" to numeric, forcing errors to NaN and filling with 0
            nadropped_roofline_data["Estimated Speedup"] = pd.to_numeric(nadropped_roofline_data["Estimated Speedup"], errors='coerce').fillna(0)

            # Create a dictionary where each kernel has a sorted list of rule descriptions
            kernel_dict = {}
            for kernel, group in nadropped_roofline_data.groupby("Kernel Name"):
                sorted_rules = sorted(set(group.sort_values("Estimated Speedup", ascending=False)["Rule Description"].tolist()), key=lambda x: group.loc[group["Rule Description"] == x, "Estimated Speedup"].max(), reverse=True)
                unique_rules = remove_similar_entries(sorted_rules)
                kernel_dict[kernel] = unique_rules

                        # Display the dictionary in Streamlit
            col1.title("Kernel Speedup Analysis")
            col1.write("### Kernel Dictionary (Sorted by Estimated Speedup)")
            col1.json(kernel_dict)
            
            # Button to submit code with roofline analysis
            if uploaded_file is not None and col2.button("Submit Code with roofline"):
                query = prepare_sources_to_submit(content=content, filtered_roofline_data=kernel_dict) 
                chatgpt_response = submit_query(query)
        except Exception as e:
            col1.error(f"Error processing Roofline data file: {e}")

    # Button to submit code with both PC sampling and importance analysis
    if uploaded_file and pc_data is not None and json_data is not None and col2.button("Submit Code with PC Sampling and importance Analysis"):
        # combined_query = f"Given the code:\n\n{content}\n\nand the following PC sampling data insights:\n"
        # for stall in find_large_stalls(pc_data, threshold=0.3):
        #     combined_query += f"Line {stall['Line No']} has high stalls due to: {', '.join(stall['Stall Reasons'])}.\n"
        # combined_query += "\nAdditionally, the following counters from the data where the values represent the affect of that particular hardware counter on runtime.\n"
        # for counter in important_counters:
        #     combined_query += f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, Value: {counter['Value']}\n"
        # combined_query += "\nThis code ran on an NVIDIA A100 machine. Can you optimize this code based on the insights provided? Do not add any explanantion just give me the optimized code only?"
        # chatgpt_response = submit_query_to_chatgpt_new(combined_query)
        query = prepare_sources_to_submit(content=content, pc_data=pc_data, important_counters=important_counters)
        chatgpt_response = submit_query(query)

    # Button to submit code with both PC sampling and roofline
    if uploaded_file and pc_data is not None and roofline_data is not None and col2.button("Submit Code with PC Sampling and roofline insights"):
        query = prepare_sources_to_submit(content=content, pc_data=pc_data, filtered_roofline_data=kernel_dict)
        chatgpt_response = submit_query(query)

    # Button to submit code with both roofline and important analysis
    if uploaded_file and json_data is not None and roofline_data is not None and col2.button("Submit Code with roofline and important analysis"):
        query = prepare_sources_to_submit(content=content, filtered_roofline_data=kernel_dict, important_counters=important_counters)
        chatgpt_response = submit_query(query)

    # Button to submit code with all sources
    if uploaded_file and pc_data is not None and json_data is not None and roofline_data is not None and col2.button("Submit Code with all sources"):
        query = prepare_sources_to_submit(content=content, pc_data=pc_data, important_counters=important_counters, filtered_roofline_data=kernel_dict)
        chatgpt_response = submit_query(query)

    if query:
        col3.subheader("ChatGPT Query")
        col3.text_area("Query:", query, height=300)

    # Display ChatGPT response
    if chatgpt_response:
        col3.subheader("ChatGPT Response")
        col3.text_area("Response:", chatgpt_response, height=600)

if __name__ == "__main__":
    main()
    # if __name__ == '__main__':
    #     if runtime.exists():
    #         main()
    #     else:
    #         sys.argv = ["streamlit", "run", sys.argv[0]]
    #         sys.exit(stcli.main())
