# for i in {0..9}
# do
#   # python save_ply.py test_result/000$i 0 0
#   # python save_ply.py test_result/000$i 0 1
#   python save_ply.py test_result/000$i 2 0
# done

for i in {10..99}
do
  # python save_ply.py test_result/00$i 0 0
  # python save_ply.py test_result/00$i 0 1
  python save_ply.py test_result/00$i 2 0
done

for i in {100..159}
do
  # python save_ply.py test_result/0$i 0 0
  # python save_ply.py test_result/0$i 0 1
  python save_ply.py test_result/0$i 2 0
done