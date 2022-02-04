load t.mat;
load p2.mat;
load GV.mat;
load Virt.dat;

a1 = -99.6;
a2 = -16;

LSPVx = -2.51;
LSPVy = -36.97;
LSPVz = 4.21;

LIPVx = -0.53;
LIPVy = -28.86;
LIPVz = 39.39;

RSPVx = -23.03;
RSPVy = 14.65;
RSPVz = -28.65; 

RIPVx = -21.90;
RIPVy = 30.13;
RIPVz = -5.53;

LAAx = 38.25;
LAAy = -17.72;
LAAz = 5.49;

%drawnow;
%trisurf(t,p2(:,1),p2(:,2),p2(:,3),GV(:,1),'FaceColor', 'interp','edgecolor','none');
trisurf(t,p2(:,1),p2(:,2),p2(:,3),Virt(100,:),'FaceColor', 'interp','edgecolor','none');
%drawnow;
 
axis equal;
daspect([1 1 1])
text(LSPVy,LSPVx,-LSPVz,' \leftarrow LSPV','FontSize',18)
text(LIPVy,LIPVx,-LIPVz,' \leftarrow LIPV','FontSize',18)
text(RSPVy,RSPVx,-RSPVz,' \leftarrow RSPV','FontSize',18)
text(RIPVy,RIPVx,-RIPVz,' \leftarrow RIPV','FontSize',18)
text(LAAy,LAAx,-LAAz,' \leftarrow LAA','FontSize',18)
view([a1 a2])
rotate3d;

return;
%%% below useless 
save t.mat t;
save p2.mat p2;
save GV.mat GV;
