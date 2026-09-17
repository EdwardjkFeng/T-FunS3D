import json
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import time

from t_funs3d.query_parsing.llm_parser import LLMParser
from t_funs3d.utils.sun3d.data_parser import DataParser
from t_funs3d.utils.misc import sort_alphanumeric


@hydra.main(version_base="1.1", config_path="config", config_name="functionality_segm")
def main(args: DictConfig):
    llm = LLMParser(model_name=args.llm.model)
    parser = DataParser(args.dataset.root, args.dataset.split)
    visits = set(sort_alphanumeric(parser.get_visits()))
    visits = sorted(list(visits))

    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)

    visits = visits[start:end]

    print(
        f"LLM processing for {end-start} visits (split {args.dataset.split}), from {visits[0]} to {visits[-1]}"
    )

    time_s = time.time()
    count_query = 0
    for visit_id in tqdm(visits):
        try:
            descs = parser.get_descriptions_list(visit_id)
            json_data = []
    
            for desc_id, query in descs.items():
                count_query += 1
    
                response = llm.parse_query(query=query)
                try:
                    if '```' in response:
                        json_dict = response.split('```')[1]
                    else:
                        json_dict = response
                    json_data.append(json_dict)
                except Exception as e:
                    print(f"error {e}")
                    json_data.append({})
    
            new_path = f"{args.dataset.root}/{args.dataset.split}/{visit_id}/{visit_id}_{args.llm.model.split('/')[-1]}_cot.json"
    
            with open(new_path, "w") as out_f:
                json.dump(json_data, out_f, indent=4)
        except:
            print(f"Visit {visit_id} does not exist")
    time_e = time.time() - time_s
    print(f"Average inference time per query is {time_e / count_query}")



if __name__ == "__main__":
    main()