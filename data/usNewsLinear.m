%% Import US News and Report data

i = 0; 
i = i + 1; a{i}.name = 'Williams'; 
a{i}.data = [1,1,100,100,93,92,3,4,1310,1530,1310,1530,90,91,19,17,3,3,71,71,3,4,7,7,96,93,1,1,97,97,6,6,3,3,57,58];
i = i + 1; a{i}.name = 'Amherst';  
a{i}.data = [2,2,98,98,91,92,3,5,1340,1540,1320,1530,87,84,15,13,5,7,73,70,3,2,8,9,94,94,1,1,98,98,9,10,2,4,58,57];
i = i + 1; a{i}.name = 'Swarthmore';
a{i}.data = [3,3,96,96,90,91,6,6,1340,1530,1350,1530,84,84,16,15,5,7,77,74,2,2,8,8,93,93,4,4,97,97,9,9,14,13,47,46];
i = i + 1; a{i}.name = 'Middlebury';
a{i}.data = [5,4,93,94,86,87,6,6,1290,1480,1290,1480,86,86,17,18,23,17,68,68,3,1,9,9,94,94,10,11,96,96,5,3,3,5,57,55];
i = i + 1; a{i}.name = 'Pomona';
a{i}.data = [4,4,94,94,87,87,1,2,1380,1560,1370,1550,91,90,15,14,20,20,71,70,1,1,7,8,94,94,3,1,98,98,8,6,19,20,43,43];
i = i + 1; a{i}.name = 'Bowdoin';
%a{i}.data = [6,6,92,93,87,87,9,8,1310,1500,1330,14902,83,83,20,16,14,14,68,68,3,1,9,10,94,93,10,6,97,96,12,14,11,9,49,50];
a{i}.data = [6,6,92,93,87,87,9,8,1310,1500,1330,1490,83,83,20,16,14,14,68,68,3,1,9,10,94,93,10,6,97,96,12,14,11,9,49,50];
i = i + 1; a{i}.name = 'Wellesley';
a{i}.data = [6,6,92,93,89,89,11,12,1270,1480,1290,1490,78,78,34,31,14,12,64,69,1,1,8,8,93,93,14,14,95,95,6,10,16,13,44,46];
i = i + 1; a{i}.name = 'Carleton';
a{i}.data = [6,8,92,92,87,88,11,12,1300,1500,1320,1510,76,78,31,31,12,16,65,65,1,1,9,9,97,97,4,4,97,97,25,27,1,2,60,58];
i = i + 1; a{i}.name = 'Haverford';
a{i}.data = [10,9,90,91,81,83,2,2,1300,1490,1300,1500,94,94,26,25,7,5,77,79,0.3,1,8,8,95,94,10,6,96,96,13,15,16,17,45,44];
i = i + 1; a{i}.name = 'Claremont McKenna';
a{i}.data = [9,10,91,90,85,85,6,14,1310,1510,1300,1480,85,71,17,14,8,4,82,86,2,2,9,9,95,94,4,11,96,96,16,21,32,20,39,43];
i = i + 1; a{i}.name = 'Vassar';
a{i}.data = [14,10,87,90,85,88,17,14,1310,1460,1320,1470,65,74,24,23,35,20,63,68,0.3,0.3,8,8,95,95,4,6,96,97,11,13,64,52,31,33];
i = i + 1; a{i}.name = 'Davidson';
a{i}.data = [11,12,89,89,84,83,11,10,1260,1430,1270,1450,78,82,29,28,8,15,70,69,0,0,10,11,99,99,4,6,96,96,37,37,7,7,54,53];
i = i + 1; a{i}.name = 'Harvey Mudd';
a{i}.data = [18,12,85,89,86,89,3,1,1410,1560,1430,1570,89,95,25,22,78,18,62,67,8,2,9,8,94,97,22,21,98,98,14,18,53,50,33,33];
i = i + 1; a{i}.name = 'US Naval Academy'; 
a{i}.data = [14,14,87,88,87,88,37,46,1140,1360,1160,1380,50,53,8,7,31,24,56,61,0.1,0,9,9,93,94,25,25,97,97,2,1,131,126,21,21];
i = i + 1; a{i}.name = 'Washington and Lee'; 
a{i}.data = [12,14,88,88,77,78,10,9,1310,1460,1310,1480,83,81,19,18,1,2,72,74,0,0.2,9,9,91,91,15,14,94,94,22,25,15,15,46,46];
i = i + 1; a{i}.name = 'Hamilton';
a{i}.data = [17,16,86,87,79,81,14,17,1300,1470,1310,1470,73,74,29,27,10,6,74,74,1,1,9,9,94,94,22,21,95,95,23,23,12,12,48,47];
i = i + 1; a{i}.name = 'Wesleyan';
a{i}.data = [12,17,88,86,85,85,14,19,1295,1480,1300,1480,68,66,21,24,44,48,67,68,5,5,9,9,97,97,4,6,96,96,25,29,9,11,50,49];
i = i + 1; a{i}.name = 'Colby';
%a{i}.data = [21,18,83,84,81,81,17,25,1260,1425,1250,14209,71,61,34,29,27,20,65,69,2,2,10,10,91,93,16,14,95,95,29,29,30,25,40,41];
a{i}.data = [21,18,83,84,81,81,17,25,1260,1425,1250,1420,71,61,34,29,27,20,65,69,2,2,10,10,91,93,16,14,95,95,29,29,30,25,40,41];
i = i + 1; a{i}.name = 'Colgate';
a{i}.data = [21,18,83,84,81,83,21,19,1270,1460,1260,1440,64,67,33,29,31,29,62,64,2,2,10,9,94,95,16,14,94,94,28,32,26,26,40,40];
i = i + 1; a{i}.name = 'Smith';
%a{i}.data = [19,18,84,84,84,85,35,35,1190,1420,1200,14402,60,61,47,45,10,20,68,66,3,5,9,9,97,97,38,35,91,92,19,21,34,36,38,36];
a{i}.data = [19,18,84,84,84,85,35,35,1190,1420,1200,1440,60,61,47,45,10,20,68,66,3,5,9,9,97,97,38,35,91,92,19,21,34,36,38,36];
%i = i + 1; a{i}.name = 'US Military Academy';
%a{i}.data = [14,18,87,84,88,87,47,42,1140,1350,1150,1370,45,48,13,11,31,36,95,94,0,0,8,8,-1,-1,50,47,92,93,1,10,83,74,28,28];
i = i + 1; a{i}.name = 'Bates';
%a{i}.data = [21,22,83,83,82,83,21,28,1240,1410,1260,14209,66,58,32,27,27,43,68,67,3,3,10,10,96,95,16,14,94,94,31,35,22,15,42,46];
a{i}.data = [21,22,83,83,82,83,21,28,1240,1410,1260,1420,66,58,32,27,27,43,68,67,3,3,10,10,96,95,16,14,94,94,31,35,22,15,42,46];
i = i + 1; a{i}.name = 'Grinnell';
a{i}.data = [19,22,84,83,85,86,38,32,1220,1470,-1,-1,62,62,43,51,14,29,68,62,0,0.3,9,9,90,91,28,28,94,94,25,27,23,27,41,40];
i = i + 1; a{i}.name = 'Macalester';
a{i}.data = [25,24,81,82,82,83,20,19,1260,1450,1240,1440,69,70,43,35,57,35,64,70,1,1,11,10,88,89,28,28,94,94,41,41,34,30,38,39];
i = i + 1; a{i}.name = 'Bryn Mawr';
a{i}.data = [25,26,81,81,81,83,32,39,1170,1420,1200,1430,64,60,48,46,35,27,74,74,4,3,8,8,90,90,38,47,93,92,19,23,26,27,40,40];
i = i + 1; a{i}.name = 'Oberlin';
a{i}.data = [24,26,82,81,81,82,16,19,1310,1470,1280,1460,69,68,31,30,27,43,73,70,3,3,9,9,95,96,35,32,94,94,31,37,41,33,36,38];


%% Convert the SAT range to be a mean, which is simply a choice to create a single number
numberOfSchools = i; 
for i=1:numberOfSchools
    a{i}.data(9) = mean(a{i}.data(9:10)); 
    a{i}.data(10) = mean(a{i}.data(11:12)); 
    a{i}.data(11:end-2) = a{i}.data(13:end);
    a{i}.data = a{i}.data(1:end-2); 
    a{i}.data13 = a{i}.data(2:2:end);
    a{i}.data12 = a{i}.data(1:2:end); 
end

%% Note, some of the variables are by rank.  So, weights will be negative as lower number means better!

for i=1:numberOfSchools
    schoolData(i,:) = a{i}.data12(3:end);  
end

%% Create a linear system with the first # of schools equalling the number
% of data points
dataPoints = length(a{1}.data12)-2; % first two data points are rank and score 
matrixSize = 20; 
A12 = zeros(matrixSize,dataPoints); 
b12 = zeros(matrixSize,1); 
for i=1:matrixSize
    A12(i,:) = schoolData(i,:);  
    b12(i) = a{i}.data12(2);   % The score the RHS
end
weights12 = A12\b12; 

%% Split data for easier future use
fid = fopen('usNewsData2012.txt','w'); 
lengthOfData = length(a{1}.data12); 
for i=1:numberOfSchools
    fprintf(fid,'%s, ',a{i}.name); 
    for j=1:4
        fprintf(fid,'%d, ',a{i}.data12(j));
    end
    fprintf(fid,'%8.3f, ',a{i}.data12(5));
    for j=6:9
        fprintf(fid,'%d, ',a{i}.data12(j));
    end    
    fprintf(fid,'%5.2f, ',a{i}.data12(10));
    for j=11:lengthOfData-1
        fprintf(fid,'%d, ',a{i}.data12(j));
    end
    fprintf(fid,'%d',a{i}.data13(lengthOfData));
    fprintf(fid,'\n'); 
end

fclose(fid);

fid = fopen('usNewsData2013.txt','w'); 
lengthOfData = length(a{1}.data13); 
for i=1:numberOfSchools
    fprintf(fid,'%s, ',a{i}.name); 
    for j=1:4
        fprintf(fid,'%d, ',a{i}.data13(j));
    end
    fprintf(fid,'%8.3f, ',a{i}.data13(5));
    for j=6:9
        fprintf(fid,'%d, ',a{i}.data13(j));
    end    
    fprintf(fid,'%5.2f, ',a{i}.data13(10));
    for j=11:lengthOfData-1
        fprintf(fid,'%d, ',a{i}.data13(j));
    end
    fprintf(fid,'%d',a{i}.data13(lengthOfData));
    fprintf(fid,'\n'); 
end

fclose(fid);

%% Columns that are by rank
rankColumns = [2 6 11 13 14];
percentOver50 = 8; 
studentFacRatio = 9;
satColumn = 3; 

% Change the matrix data so larger entries get a larger scale
schoolDataAltered = schoolData; 
schoolDataAltered(:,rankColumns) = 100 - schoolData(:,rankColumns); 
schoolDataAltered(:,percentOver50) = 1./(1+schoolData(:,percentOver50));
schoolDataAltered(:,studentFacRatio) = 1./schoolData(:,studentFacRatio);

A12altered = schoolDataAltered(1:matrixSize,:); 
weights12new = A12altered\b12; 

%% Given the linear weights predict a school's result

fprintf('\n=====================================================================\n'); 
fprintf('              School   Actual Score   1st prediction     2nd prediction\n'); 
fprintf('=======================================================================\n'); 
%for i=matrixSize+1:numberOfSchools
for i=1:numberOfSchools
    fprintf('%20s   %6.0f           %6.2f            %6.2f\n',a{i}.name,a{i}.data12(2),schoolData(i,:)*weights12,schoolDataAltered(i,:)*weights12new); 
end
 

