'''
    문제 설명
    자연수 n이 주어졌을 때, n의 다음 큰 숫자는 다음과 같이 정의 합니다.

    조건 1. n의 다음 큰 숫자는 n보다 큰 자연수 입니다.
    조건 2. n의 다음 큰 숫자와 n은 2진수로 변환했을 때 1의 갯수가 같습니다.
    조건 3. n의 다음 큰 숫자는 조건 1, 2를 만족하는 수 중 가장 작은 수 입니다.
    예를 들어서 78(1001110)의 다음 큰 숫자는 83(1010011)입니다.

    자연수 n이 매개변수로 주어질 때, n의 다음 큰 숫자를 return 하는 solution 함수를 완성해주세요.

    제한 사항
    n은 1,000,000 이하의 자연수 입니다.
'''


# TODO : 정확성 35.0 / 효율성 15.0 ---- 50.0 / 50점  ===> 25분 소요
n = 15

def solution(n):
    answer = 0
    # 자연수를 2진수로 바꾸는 format 함수
    b = format(n, 'b')
    # n보다 크지만 가장 작은 수의 1의 갯수가 같은 값을 찾기 위한 변수 선언
    b_copy = format(n+1, 'b')

    while True:
        if b_copy.count('1') == b.count('1'):
            # 이진수를 십진수로 변환
            answer = int(b_copy, 2)
            break
        else:
            # format 함수 내에서 b_copy 수를 1씩 추가.. ( 굳이..? )
            b_copy = format(int(b_copy, 2) + 1, 'b')
            print(b_copy)
    print(answer)

############################################
# 구글링 간단한 풀이 ..
############################################

def solution(n):
    c = n+1
    while True:
        if bin(c).count('1') == bin(n).count('1'):
            return c
        c += 1

'''
방금그곡
'''

m = 'ABCDEFG'
musicinfos = ["12:00,12:14,HELLO,CD#EFGAB", "13:00,13:05,WORLD,ABCDEF"]

spl = list(musicinfos[0].split(',')[-1])

print(spl)

if '#' in spl:
    pass

# 못 풂

# 1. C#, A# 같은 값을 하나로 치환해줘야함 -> mapping , lambda , replace 사용
# 2. musicinfos 리스트의 각 인덱스 값을 구분해주어야 함. -> split(',') 사용
# 3. 시간 구분해줘야함. -> mapping, datetime 등의 방법 사용
# 4. m의 길이와 musicinfos 안의 재생시간을 비교 -> 길이가 시간보다 긴 경우 / 짧은 경우
#    짧다면 시간을 계산해서 몫, 나머지로 계산
# 5. 포함된다면 if~ in 사용 / 빈 리스트 생성해서 append // 없다면 'None' 출력