import csv

# 파일 경로
input_file_path = 'number_6500_20.csv'
output_file_path = 'add_to.csv'

# 숫자 입력을 위한 변수 초기화
numbers = list(range(65))

# 파일 열기 및 숫자 입력
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    data_rows = []
    for i in range(65):
        data_rows.extend([[i]] * 100)

    for i in range(6500):
        writer.writerow(data_rows[i])

print('숫자 입력이 완료되었습니다.')
