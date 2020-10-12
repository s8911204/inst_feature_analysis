#!/usr/bin/python3
import sys
import os
import json
import subprocess


def parse_and_save(debug_path, output_folder):

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
        if l.startswith('# End machine code for function ') and func_name is not None and recording:
            print('Total lines %s -- %s : %d' % (func_name, pass_name, len(mcode_lines)))
            recording = False
            escaped_passname = pass_name.replace(' ', '-').replace('&', '-and-').replace('/', '-or-').replace("'","").replace('"','')
            print(escaped_passname)
            code_path = os.path.join(output_folder, '%s_%s.ir' % (func_name, escaped_passname))
            with open(code_path, 'w') as ofile:
                ofile.write('\n'.join(mcode_lines))
            mcode_lines.clear()
            info_summary[func_name].append({'pass': pass_name, 'path': code_path})
    with open(os.path.join(output_folder, 'info_summary.json'), 'w') as sfile:
        sfile.write(json.dumps(info_summary, indent=4))

def diff(output_folder, diff_folder):
    with open(os.path.join(output_folder, 'info_summary.json'), 'r') as sfile:
        summary = json.load(sfile)
    for f in summary.keys():
        print('Passes which changed :%s' % f)
        f_info = summary[f]
        for i in range(0, len(f_info)):
            if i == 0:
                prev_ir = f_info[i]['path']
                prev_pass = f_info[i]['pass']
            else:
                base_ir = f_info[i]['path']
                base_pass = f_info[i]['pass']
                # print('Start diff %s and %s' % (prev_pass, base_pass))
                cmd = 'diff %s %s > tmp_diff.txt' % (prev_ir, base_ir)
                ret_code = subprocess.call(cmd, shell=True)
                if ret_code == 1:
                    print('    %s' % base_pass)
                    escaped_passname = base_pass.replace(' ', '-').replace('&', '-and-').replace('/', '-or-').replace("'","").replace('"','')
                    diff_path = os.path.join(diff_folder, 'diff_%d_%s_%s' % (i, f, escaped_passname))
                    cmd = 'mv tmp_diff.txt %s' % diff_path
                    subprocess.call(cmd, shell=True)
                    f_info[i]['change'] = True
                    f_info[i]['diff'] = diff_path
                    prev_ir = base_ir
                    prev_pass = base_pass
    with open(os.path.join(output_folder, 'info_summary.json'), 'w') as ofile:
        ofile.write(json.dumps(summary, indent=4))


def main():
    debug_path = sys.argv[1]
    output_folder = sys.argv[2]
    diff_folder = sys.argv[3]
    parse_and_save(debug_path, output_folder)
    diff(output_folder, diff_folder)


if __name__ == '__main__':
    main()