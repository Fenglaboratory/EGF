import itertools
import argparse
import requests
import tqdm
import json


def map_label_to_auth_id(label_tag):
    # Assume pdb_id is the ID you got from your search
    entry_id = label_tag[:4]
    asym_id = label_tag[5:]

    # Make a request to the PDB API
    response = requests.get(
        f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{entry_id}/{asym_id}"
    )

    # Parse the JSON response
    data = response.json()

    auth_chain = data["rcsb_polymer_entity_instance_container_identifiers"][
        "auth_asym_id"
    ]

    return f"{entry_id}_{auth_chain}"


search_json = """{
  "query": {
    "type": "terminal",
    "service": "sequence",
    "parameters": {
      "evalue_cutoff": 1,
      "identity_cutoff": 0.9,
      "sequence_type": "protein",
      "value": "SEQUENCE_HERE"
    }
  },
  "request_options": {
    "scoring_strategy": "sequence",
    "paginate": {
      "start": 0,
      "rows": 1000
    }
  },
  "return_type": "polymer_instance"
}"""


def search_for_similar_proteins(chains, sequence_dict):
    all_search_results = {}
    for i, chain in enumerate(tqdm.tqdm(chains)):
        search_query = search_json.replace("SEQUENCE_HERE", sequence_dict[chain])
        endpoint = f"https://search.rcsb.org/rcsbsearch/v2/query?json={search_query}"
        result = requests.get(endpoint)
        all_search_results[chain] = result.content.decode("utf-8")
    return all_search_results


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag_cluster_mapping", type=str, required=True)
    parser.add_argument("--sequence_dict_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.tag_cluster_mapping, "r") as f:
        tcm = json.load(f)

    with open(args.sequence_dict_path, "r") as f:
        sequence_dict = json.load(f)

    tcm_values = list(itertools.chain(*tcm.values()))
    all_search_results = search_for_similar_proteins(tcm_values, sequence_dict)

    processed_search_results = {}
    for tag, results_string in tqdm.tqdm(
        all_search_results.items(), total=len(all_search_results)
    ):
        result_set = [
            x["identifier"].replace(".", "_")
            for x in json.loads(results_string)["result_set"]
        ]
        processed_search_results[tag] = result_set

    final_search_results = {}
    for tag, results in tqdm.tqdm(processed_search_results.items()):
        new_results = [map_label_to_auth_id(x) for x in results]
        final_search_results[tag] = new_results

    with open(args.output_path, "w") as f:
        json.dump(final_search_results, f)


if __name__ == "__main__":
    main()
