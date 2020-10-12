#!/usr/bin/python3
import sys
import os
import json


def main():
    debug_path = sys.argv[1]
    output_folder = sys.argv[2]
    lines = []
    with open(debug_path, 'r') as rfile:
        for line in rfile:
            lines.append(line.replace('\n', ''))
    mcode_lines = []
    info_summary = {}
    recording = False
    func_name = None
    for i in range(0, len(lines)):
        l = lines[i]
        if l.startswith('# Machine code for function') and\
                lines[i - 1].startswith('# *** IR Dump After '):
            pl = lines[i - 1]
            pass_name = pl.replace('# *** IR Dump After', '').replace('***:', '').strip()
            tmp0 = l.replace('# Machine code for function', '').lstrip().split(':')
            func_name = tmp0[0]
            if func_name not in info_summary:
                info_summary[func_name] = []
            mcode_lines.clear()
            mcode_lines.append(tmp0[1].strip())
            print('Start recording %s --- %s' % (func_name, pass_name))
            recording = True
        if recording:
            mcode_lines.append(l)
        if l.startswith('# End machine code for function ') and func_name is not None:
            print('Total lines %s -- %s : %d' % (func_name, pass_name, len(mcode_lines)))
            recording = False
            code_path = os.path.join(output_folder, '%s_%s.ir' % (func_name, pass_name))
            with open(code_path, 'w') as ofile:
                ofile.write('\n'.join(mcode_lines))
            mcode_lines.clear()
            info_summary[func_name].append({'pass': pass_name, 'path': code_path})
    with open(os.path.join(output_folder, 'info_summary.json'), 'w') as sfile:
        sfile.write(json.dumps(info_summary, indent=4))


if __name__ == '__main__':
    main()