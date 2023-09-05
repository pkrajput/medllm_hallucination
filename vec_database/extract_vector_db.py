from knowledge_extractor import KnowledgeExtractor
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--query",
    default=None,
    type=str,
)
parser.add_argument(
    "--passback_gpt",
    default=False,
    type=bool,
)
parser.add_argument(
    "--number_of_matches",
    default=3,
    type=int,
)

if __name__ == "__main__":

    args = parser.parse_args()

    ke = KnowledgeExtractor()
    query = args.query
    query_result = ke.get_related_knowledge(
        query, top_k=args.number_of_matches, passback_gpt=args.passback_gpt
    )

    print(query_result)
