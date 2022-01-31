@ECHO OFF
rem Wrapper for CGCNN
call conda activate cgcnn
if %1.==. (
	set ini=init.ini
) else (
	set ini=%1
)
set /p name= <%ini%
set name=%name:~3%
echo %name%
for /f "tokens=*" %%g in ('fargp.py %ini%') do set params=%%g
echo %params%
python main.py %params% > %name%.out

ren test_results.csv %name%.csv
ren losses_train.txt %name%_train.txt
ren losses_val.txt %name%_val.txt
ren losses_test.txt %name%_test.txt
ren model_best.pth.tar %name%.pth.tar
Rscript Lossplot.R %name% >> %name%.out
ren Rplots.pdf %name%.pdf

python predict.py %name%.pth.tar %params%
ren test_results.csv %name%_p.csv

call conda deactivate
PAUSE