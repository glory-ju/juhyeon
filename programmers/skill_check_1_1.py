'''
    소수 만들기 문제
    -> nums 리스트에서 3개의 숫자만을 조합하여 합한 것이 소수인 경우의 수를 구하기
'''


    # TODO: 1번 풀이 ( itertools 라이브러리 사용 )
    # 풀이시간 : 11분


from  itertools import combinations

nums = [1,2,3,4] # result 1
nums = [1,2,7,6,4] # result 4

def solution_1(nums):
    result = 0

    # nums 리스트에서 3개를 중복 허용하지 않고 뽑은 인수들의 합을 저장할 빈 리스트 선언
    sum_list = []
    comb = list(combinations(nums, 3))
    # sum_list에 조합(combination)을 사용하여 합한 결괏값들 저장
    for idx in comb:
        sum_list.append(sum(idx))

    # sum_list 안의 인자들이 소수인지 아닌지 판별하는 for문
    for i in sum_list:
        # 인자가 넘어갈 때마다 answer를 0으로 초기화
        answer = 0
        for j in range(1, i+1):
            # 소수는 약수가 자기 자신과 1, 두 개 뿐이기 때문에 answer가 2인 경우 result 갯수 1씩 증가
            if i % j == 0:
                answer += 1
        if answer == 2:
            result += 1

    return result
print(solution_1(nums))

    # TODO: 2번 풀이 ( 라이브러리 미사용 )
    # 풀이시간 : 15분

nums = [1,2,3,4] # result 1
nums = [1,2,7,6,4] # result 4

def solution_2(nums):
    result = 0

    # combination 하는 3중 for문
    for i in range(len(nums)-2):
        for j in range(i+1, len(nums)-1):
            for k in range(j+1, len(nums)):
                answer = 0
                sumation = nums[i] + nums[j] + nums[k]

                print(f'sumation:', nums[i], nums[j], nums[k], '=', sumation)

                # 시간 복잡도 때문에 나중에는 에라토스테네스의 체 활용하는 것이 나아보임.(math.sqrt() 사용)
                for idx in range(1, sumation+1):
                    if sumation % idx == 0:
                        answer += 1
                print(f'answer:', answer)
                if answer == 2:
                    result += 1
    return result
print(solution_2(nums))