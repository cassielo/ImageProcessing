close all;
clear all;
clc;
% Read image
Q1 = imread('Q1.tif');
Q2 = imread('Q2.tif');
Q3 = imread('Q3.tif');
Q4 = imread('Q4.tif');
% do (a) and (b) for each image
qa(Q1)
qb(Q1)
qa(Q2)
qb(Q2)
qa(Q3)
qb(Q3)
qa(Q4)
qb(Q4)

%(a) using H(u, v) obtained from h(x, y)
function qa(pic)
    % Initialize
    I = mat2gray(pic,[0 255]);
    I = I(:,:,1);
    [M,N] = size(I);
    
    % Fourier transform of the original image
    F = fft2(I);
    
    % Create h(x, y) (PSF)
    LEN = sqrt((0.1*M)^2+(0.1*N)^2);
    THETA = -atan(M/N)/pi*180;
    PSF = fspecial('motion',LEN,THETA);
    
    % Fourier transform of h(x,y) (=H(u,v))
    H1 = psf2otf(PSF,[M N]);
    
    % Multiply two transforms (=G)
    G1 = H1 .* F;
    
    % Take the inverse transform of G
    g1 = real(ifft2(G1));
    g1 = circshift(g1, [round(0.1*M/2),round(0.1*N/2)]);
    
    % Show the blurred image
    figure();
    imshow(g1,[0 1]);
    title('(a)');

end

%(b) sampling H(u, v) in Slide 32
function qb(pic)
    % Initialize
    f = mat2gray(pic,[0 255]);
    f2 = f(:,:,1);
    [P,Q] = size(f2);
    
    % Make the side lengths to be even
    if mod(P,2)~=0
        P = P-1;
    end
    if mod(Q,2)~=0
        Q = Q-1;
    end
    f2 = f2(1:1:P,1:1:Q);
    
    % Multiply by (-1)^(x+y) to center its transform
    for x = 1:1:P
        for y = 1:1:Q
            fc(x,y) = f2(x,y) * (-1)^(x+y);
        end
    end
    
    % Fourier transform of the image
    F_I = fft2(fc);
    
    % Sample H(u, v) in Slide 32
    H = zeros(P,Q);
    a = 0.1;
    b = 0.1;
    T = 1;
    for x = (-P/2):1:(P/2)-1
         for y = (-Q/2):1:(Q/2)-1
            R = (x*a + y*b)*pi;
            if(R == 0)
                H(x+(P/2)+1,y+(Q/2)+1) = T;
            else
                H(x+(P/2)+1,y+(Q/2)+1) = (T/R)*(sin(R))*exp(-1i*R);
            end
         end
    end
    
    % Multiply two transforms (=G)
    G = H .* F_I;
    
    % Take the inverse transform of G
    gc = ifft2(G);
    
    % Obtain the processed image
    for x = 1:1:(P)
        for y = 1:1:(Q)
            g(x,y) = real(gc(x,y)) * (-1)^(x+y);
        end
    end
    
    % Show the blurred image
    figure();
    imshow(g,[0 1]);
    title('(b)');
end





