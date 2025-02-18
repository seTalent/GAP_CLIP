
train_set = "annotation/DFEW_train_set1.txt"
test_set = "annotation/DFEW_test_set1.txt"

def post_process_annotation(file_path):
    new_lines = []
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            data = line.split(" ")
            _file_path = data[0].zfill(5)
            frame_cnt = data[1]
            label = data[2]

            label = int(label) - 1

            _file_path = "data/DFEW/frames/" + _file_path

            new_line = _file_path + " " + frame_cnt + " " + str(label)
            new_lines.append(new_line)



    with open(file_path, "w", encoding='utf-8') as f:
        for line in new_lines:
            f.write(line +'\n')


if __name__ == '__main__':
    post_process_annotation(train_set)
    post_process_annotation(test_set)
