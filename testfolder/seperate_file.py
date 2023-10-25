import csv

input_file_path = 'only_onemore_angle_fy_6500_20_beforseperate_2.csv'

# 파일 분할 조건
ranges = [(0, 69), (70, 89), (90, 99)]
output_files = ['train_add_data.csv', 'valid_add_data.csv', 'test_add_data.csv']

# 파일 열기
with open(input_file_path, 'r') as input_file:
    reader = csv.reader(input_file)

    # 출력 파일 열기
    output_writers = []
    for output_file in output_files:
        output_writers.append(open(output_file, 'w', newline=''))

    # 행 분할 및 저장
    for row_index, row in enumerate(reader):
        for index, (start, end) in enumerate(ranges):
            if start <= row_index % 100 <= end:
                output_writers[index].write(','.join(row) + '\n')
                break

    # 출력 파일 닫기
    for output_writer in output_writers:
        output_writer.close()

print('파일 분할이 완료되었습니다.')
