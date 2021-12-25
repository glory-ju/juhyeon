'''
    문제 설명
    JadenCase란 모든 단어의 첫 문자가 대문자이고, 그 외의 알파벳은 소문자인 문자열입니다.
    문자열 s가 주어졌을 때, s를 JadenCase로 바꾼 문자열을 리턴하는 함수, solution을 완성해주세요.
'''

# TODO: 1번 풀이
# 풀이시간 : 11분 => 21.9 / 50점
# 런타임 에러

s = "3people unFollowed me"
return_s = "3people Unfollowed Me"
s = "3for the last week"
return_s = "For The Last Week"

def solution(s):
    answer = ''
    s_split = s.split(' ')

    for i in range(len(s_split)):
        sentence = s_split[i][0].upper() + s_split[i][1:].lower() + ' '
        answer += sentence
    return answer.rstrip()

# TODO: 2번 풀이
# 풀이시간 5분 => 6.3 / 50점

s = "3people unFollowed me"
return_s = "3people Unfollowed Me"
# s = "3for the last week"
return_s = "For The Last Week"

def solution(s):
    answer = ''
    s_split = s.lower().split()

    for i in range(len(s_split)):
        before = s_split[i][0]
        after = s_split[i].replace(before, before.upper())
        answer += ' ' + after
    return answer.lstrip()