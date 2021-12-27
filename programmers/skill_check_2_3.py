'''
    ##### 문제 1

    프렌즈4블록
    블라인드 공채를 통과한 신입 사원 라이언은 신규 게임 개발 업무를 맡게 되었다. 이번에 출시할 게임 제목은 "프렌즈4블록".
    같은 모양의 카카오프렌즈 블록이 2×2 형태로 4개가 붙어있을 경우 사라지면서 점수를 얻는 게임이다.

    위 초기 배치를 문자로 표시하면 아래와 같다.

    TTTANT
    RRFACC
    RRRFCC
    TRRRAA
    TTMMMF
    TMMTTJ
    각 문자는 라이언(R), 무지(M), 어피치(A), 프로도(F), 네오(N), 튜브(T), 제이지(J), 콘(C)을 의미한다

    입력으로 블록의 첫 배치가 주어졌을 때, 지워지는 블록은 모두 몇 개인지 판단하는 프로그램을 제작하라.
'''

# TODO: 입력으로 주어진 판 정보를 가지고 몇 개의 블록이 지워질 지 출력하라.
# 풀이시간 25분 // 해결 못함..

m_1 = 4
n_1 = 5
board_1 = ["CCBDE",
           "AAADE",
           "AAABF",
           "CCBBF"]
answer_1 = 14

m_2 = 6
n_2 = 6
board_2 = ["TTTANT",
           "RRFACC",
           "RRRFCC",
           "TRRRAA",
           "TTMMMF",
           "TMMTTJ"]
answer_2 = 15

def solution(m_1, n_1, board_1):
    answer = 0

    # 연속되는 2개의 문자가 같을 경우를 구현해야 하는데
    # for문 안에서 인덱스 +1 을 하기 위해선 range 값에 -1을 해줘야 함.
    for i in range(m_1-1):
        for j in range(n_1-1):
            # if board_1[i][j] + board_1[i][j+1] == board_1[i+1][j] + board_1[i+1][j+1]:
            #     board_1[i] = board_1[i].strip(board_1[i][j]+board_1[i][j+1])
            # i번째 인덱스에서 j번째 element와 j+1번째 element가 같다면
            if board_1[i][j] == board_1[i][j+1]:
                # i+1번째 인덱스에서 j번째 element와 j+1번째 element가 같은 값이 i번째 인덱스에서의 j번째 값과 같다면
                if board_1[i+1][j] == board_1[i+1][j+1] == board_1[i][j]:
                    # board_1[i].strip(이 값을 채워야 하는데 생각이 좀 필요함.)
                    board_1[i] = board_1[i].strip()
                    ##### 여기까지밖에 구현 못했음. 시간 남을 때 해결할 것.
                    print(board_1)

    return answer
# solution(m_1,n_1,board_1)

'''
    ##### 문제 2
    
    후보키
'''
# TODO: 학생들의 인적사항이 주어졌을 때, 후보 키의 최대 개수를 구하라.

relation = [["100","ryan","music","2"],
            ["200","apeach","math","2"],
            ["300","tube","computer","3"],
            ["400","con","computer","4"],
            ["500","muzi","music","3"],
            ["600","apeach","music","2"]]

from collections import Counter
def solution(relation):
    answer = 0
    # for i, element in enumerate(relation):
    #     for j in range(len(element)):
    #         if relation[i][j].count(element[j]) == 1:
    #             answer += 1
    #             print(answer)
    #         else: pass

    # relation 리스트의 여러 개의 객체를 포함하고 있는 하나의 객체를 풀어주는 방법 --> unpacking :  *relation
    relation_list = list(zip(*relation))
    # -> 리스트 안의 같은 axis 값들이 다시 묶임. -> [('100','200','300', ... ), ('ryan', 'apeech', ...)]
    for i in range(len(relation)):
        counter = Counter(relation[i])
        print(counter)

        # unpacking한 리스트에서 most_common 값이 1일 경우 answer += 1
        # 2 이상일 경우에는 두 객체를 zipping 해서 찾아보는 과정을 연구해야 함.
        # 구현할 시간이 필요해 보임.

# solution(relation)
# print(relation[:4][:2])
relation_list = list(zip(*relation))
counter = Counter(relation_list[1])
# print(counter.most_common(1)[0][1])
