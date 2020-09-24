import os

###########################################################################
# TODO: 파일명을 수정하시오                                                  #
###########################################################################
outfile_name = "과제4_20123456_홍길동"

# 과제 제출 파일 리스트
file_list = [
    './nn/activation.py',
    './nn/conv.py',
    './nn/optimizer.py',
    './nn/maxpooling.py',
    'mnist_test.py'
]

# anaconda와 두가지 패키지가 필요합니다.
os.system("pip install ipynb-py-convert")
os.system("pip install nbmerge")

###########################################################################
# submission 생성 코드                                                     #
###########################################################################

# 파일별 데이터 모으기
for file in file_list:
    os.system("ipynb-py-convert " +file+" "+os.path.splitext(os.path.basename(file))[0]+".ipynb")

# 파일 리스트 모아서 한개의 notebook 만들기
merge_list = ""
for file in file_list:
    merge_list += os.path.splitext(os.path.basename(file))[0]+".ipynb "
os.system("nbmerge " +merge_list+ " > "+outfile_name+".ipynb")

# 노트북 파일 pdf로 바꾸기
os.system("jupyter nbconvert " +outfile_name+ ".ipynb")

# 생성한 notebook 모두 삭제
file_list.append(outfile_name+ ".ipynb")
for file in file_list:
    ntfile = os.path.splitext(os.path.basename(file))[0]+".ipynb"
    if os.path.exists(ntfile):
        os.remove(ntfile)
    else:
        print("The file does not exist")



