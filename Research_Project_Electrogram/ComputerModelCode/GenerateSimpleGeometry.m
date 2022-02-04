%% Setting up atrial geometry
% this file creates the file "Input2Datrium.dat" which
% will be used as a template for solving the propagation patterns
% numerically
Nx = 500;
Ny = 500;
Nz = 5;

AtriaTissue = zeros([Ny Nx Nz],'single');
AtriaTissue(:, :, :) = 1;
AtriaTissue = logical(AtriaTissue);
[Ny, Nx, Nz] = size(AtriaTissue);

needsave = 1;
if needsave ==1,
   %save tissue 
   fid = fopen('Input2Datrium.dat', 'a');
   %fid = fopen('InputFibrosis2.dat', 'a');
   for plane=1:Nz,
       for Yi=1:Ny,
           for Xi=1:Nx,
               %%% tissue part
               fprintf(fid, '%d',AtriaTissue(Yi, Xi, plane)) ;
           end;
	   fprintf(fid, '\n');
       end;
	fprintf(fid, '\n');
   end;
   fclose(fid);
end;

fprintf('\nThe size of the atrial geometry dimension generated is Ny=%d, Nx=%d, Nz=%d\n\n',Ny,Nx,Nz);