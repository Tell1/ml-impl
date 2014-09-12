function tam = regiones_voronoi(clases, centros)

% Función que devuelve el tamaño de las regiones de voronoi calculadas en
% k-medias.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

tam = zeros(1,size(centros, 1));

for i=1:size(clases, 1)
   tam(clases(i)) = tam(clases(i))+1;
end