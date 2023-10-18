#!/usr/bin/python3
import argparse
import json
import logging
import os
from shutil import copyfile

import utils

logging.basicConfig(
    filename="logfile.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

ELFCRTSymbols = [
    "call_weak_fn",
    "deregister_tm_clones",
    "__do_global_dtors_aux",
    "__do_global_dtors_aux_fini_array_entry",
    "_fini",
    "frame_dummy",
    "__frame_dummy_init_array_entry",
    "_init",
    "__init_array_end",
    "__init_array_start",
    "__libc_csu_fini",
    "__libc_csu_init",
    "register_tm_clones",
    "_start",
    "_dl_relocate_static_pie",
]


def get_args():
    logging.info("Starting function get_args")
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="the folder which contains dataset.")
    parser.add_argument(
        "--out_dir", required=True, help="the folder which used as output base folder."
    )
    args = parser.parse_args()
    logging.info("Ending function get_args with args: %s", args)
    return args


def show_diff(df1, df2):
    logging.info("Starting function show_diff with df1: %s, df2: %s", df1, df2)
    try:
        print(df1.ne(df2).any(1))
    except:
        print("# of instrs: %d %d" % (len(df1), len(df2)))
    logging.info("Ending function show_diff")


def show_crt_summary(summary, crt):
    logging.info(
        "Starting function show_crt_summary with summary: %s, crt: %s", summary, crt
    )
    crt_summary = summary[crt]
    list = []
    for key in crt_summary.keys():
        if key == "current_key":
            continue
        else:
            repr = key
            count = crt_summary[key]["count"]
            fullpath = crt_summary[key]["path"]
            list.append({"represent": repr, "count": count, "path": fullpath})
    print("%s: %s" % (crt, json.dumps(list, indent=4)))
    logging.info("Ending function show_crt_summary with list: %s", list)
    return list


def patterns_output(samples, output_base):
    logging.info(
        "Starting function patterns_output with samples: %s, output_base: %s",
        samples,
        output_base,
    )
    info_lines = []
    for crt in samples.keys():
        seq = 0
        sorted_pat = sorted(samples[crt], key=lambda x: x["count"], reverse=True)
        for pat in sorted_pat:
            file_name = "%s__%d.csv" % (crt, seq)
            out_path = os.path.join(output_base, file_name)
            src_path = pat["path"]
            copyfile(src_path, out_path)
            line = "%s,%s" % (crt, out_path)
            info_lines.append(line)
            seq += 1
    with open(os.path.join(output_base, "info_csv"), "w") as info_csv:
        info_csv.write("\n".join(info_lines))
    logging.info("Ending function patterns_output with info_lines: %s", info_lines)


def main():
    logging.info("Starting function main")
    arg = get_args()
    summary = {}
    samples_stat = {}
    for crt in ELFCRTSymbols:
        base = None
        summary[crt] = {}
        print("##### %s ######" % crt)
        for f in os.listdir(arg.data_folder):
            full_path = os.path.join(arg.data_folder, f)
            if not os.path.isfile(full_path):
                continue
            if not f.startswith("CRT_999_" + crt):
                continue
            key = f.split("_999_")[2].replace(".csv", "")
            if base is None:
                base = utils.toDf(full_path)
                summary[crt][key] = {}
                summary[crt][key]["count"] = 1
                summary[crt][key]["df"] = base
                summary[crt][key]["path"] = full_path
                summary[crt]["current_key"] = key
                continue
            data = utils.toDf(full_path)
            base_df = base.drop(columns=["module name", "target"])
            data_df = data.drop(columns=["module name", "target"])
            pattern_found = False
            if not base_df.equals(data_df):
                show_diff(base_df, data_df)
                for old_key in summary[crt].keys():
                    if old_key == "current_key":
                        continue
                    pattern_df = summary[crt][old_key]["df"].drop(
                        columns=["module name", "target"]
                    )
                    if pattern_df.equals(data_df):
                        pattern_found = True
                        base = summary[crt][old_key]["df"]
                        summary[crt]["current_key"] = old_key
                        summary[crt][old_key]["count"] += 1
                        break
                if not pattern_found:
                    base = data
                    summary[crt][key] = {}
                    summary[crt][key]["count"] = 1
                    summary[crt][key]["df"] = base
                    summary[crt][key]["path"] = full_path
                    summary[crt]["current_key"] = key
            else:
                summary[crt][summary[crt]["current_key"]]["count"] += 1
        samples_stat[crt] = show_crt_summary(summary, crt)
    patterns_output(samples_stat, arg.out_dir)
    logging.info("Ending function main")


if __name__ == "__main__":
    main()
