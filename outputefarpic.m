REAL(REAL>threshold)=1; 
REAL(REAL==threshold)=0; 
pcolor(REAL);
 a = max(max(REAL));
shading interp
colorbar
surf(REAL);
shading interp