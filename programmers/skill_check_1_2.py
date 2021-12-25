'''
    다소 너무 쉬웠던 문제...
    다른 문제를 풀어보려고 했으나 이미 level 1은 pass 를 받아서
    풀 수 없었음.
    풀이 시간 : 20초
'''


def solution(num):
    answer = ''
    if num % 2 == 0: answer = 'Even'
    else: answer = 'Odd'
    return answer