import csv

# 입력 파일 이름과 출력 파일 이름
input_file = 'testdata_fy.csv'
output_file = 'test_modified.csv'

# 입력 파일 열기
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# 모든 음수 값을 양수 값으로 변경
for i in range(len(data)):
    for j in range(len(data[i])):
        value = float(data[i][j])
        if value < 0:
            data[i][j] = str(-value)

# 출력 파일 저장
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Modified data saved to '{output_file}'.")