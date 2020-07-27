import sys, os
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("python data_dir input_name output1_name output2_name ratio(1/(1+2))")
        sys.exit(0)
    data_dir = sys.argv[1]
    input_name = sys.argv[2]
    output1_name = sys.argv[3]
    output2_name = sys.argv[4]
    ratio = float(sys.argv[5])

    input_path = os.path.join(data_dir, input_name)
    output1_path = os.path.join(data_dir, output1_name)
    output2_path = os.path.join(data_dir, output2_name)

    with open(output1_path, 'w', encoding='utf-8') as fw1:
        with open(output2_path, 'w', encoding='utf-8') as fw2:
            with open(input_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    r = np.random.random()
                    if r < ratio:
                        fw1.write(line)
                    else:
                        fw2.write(line)
                print("Write done! %s", output1_path)

