#!/usr/bin/python3
import logging
import os
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler()],
)

target_folder = "/home/mtk02470/temp/my"


def main():
    logging.info("Starting function main")
    for f in os.listdir(target_folder):
        fpath = os.path.join(target_folder, f)
        if os.path.isfile(fpath):
            logging.info("Deleting file: %s", fpath)
            os.remove(fpath)
    dot_files = []
    for file in os.listdir("/tmp"):
        if file.endswith(".dot") and file.startswith("dag."):
            target_path = os.path.join(target_folder, file)
            logging.info("Moving file: %s to %s", file, target_path)
            shutil.move(os.path.join("/tmp", file), target_path)
            dot_files.append(target_path)
    for dotf in dot_files:
        pngf = dotf.replace(".dot", ".png")
        cmd = "dot -Tpng %s > %s" % (dotf, pngf)
        logging.info("Executing command: %s", cmd)
        os.system(cmd)
    logging.info("Ending function main")


if __name__ == "__main__":
    main()
