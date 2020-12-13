#!/usr/bin/python3
import os
import shutil

target_folder = '/home/mtk02470/temp/my'


def main():
    for f in os.listdir(target_folder):
        fpath = os.path.join(target_folder, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    dot_files = []
    for file in os.listdir('/tmp'):
        if file.endswith('.dot') and file.startswith('dag.'):
            target_path = os.path.join(target_folder, file)
            shutil.move(os.path.join('/tmp', file), target_path)
            dot_files.append(target_path)
    for dotf in dot_files:
        pngf = dotf.replace('.dot', '.png')
        cmd = 'dot -Tpng %s > %s' % (dotf, pngf)
        print(cmd)
        os.system(cmd)

    
if __name__ == '__main__':
    main()