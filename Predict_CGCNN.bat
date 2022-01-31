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
for /f "usebackq delims=" %%a in (`more +1 %ini%`) DO (
  set model=%%a
  goto :leave
)
:leave
set model=%model:~3%

echo %name%
echo %model%

for /f "tokens=*" %%g in ('fargp.py %ini%') do set params=%%g
echo %params%

python predict.py %model%.pth.tar %params%
ren test_results.csv %name%_p.csv

call conda deactivate
pause
