% This matlab file create a dataset to define MRF energy function, and by using a MRF solver as a black box
% find the minimum set of labels in order to register 2 images (non-rigid registration)
% The MRF used is a free-form deformation grid


% The MRF solver needs 8 datas to work :
% - num_nodes : the total number of nodes (= node_grid_size*node_grid_size) , num_nodes >=4
% - num_labels : the numbers of labels in a node's label set (= label_grid_size*label_grid_size)
% - num_edges : the number of edges of the grid (= 2*node_grid_size*(node_grid_size-1) )
% - num_max_iters : number of maximum iterations of the algorith
% - edges : list of size 2 * num_edges containing edges, i-th egde is defined by edges(2*i-1) - edges (2*i)
% - wcosts : list of weights associated to each edges, in our case, a list full of ones
% - uPotential : list of unary potenatial (data term that encode likelyhood) of size num_nodes * num_label , the k-th label of the i-th node is defined by : uPotential( (i-1)*num_label + k )
% - pPotential : list of pairwise potential (regularization term the encode geometric constraints) , pPotential(label_i, label_j) = pPotential( (j-1)*num_labels + i)


% dataset is written in a binary file, then go through the mrf solver, then the output is used in matlab to get the deformation field 


% this is the main
% list of parameters :
% - fixed_path : path to target image
% - moving_path : path to source image
% - node_grid_sizes : list of 2 elements containing node grid sizes for each dimension (first element is height, second is width
% - pyramidLevels : numver of pyramid levels to perform registration. if 1 then registration is perfom only once with node_grid_sizes parameters, else node grid sizes is double at each level.
% - iterations : number of iterations of the algorithm for each pyramidLevels
% - label_grid_size : size of the label space
% - num_max_iters : number of maximum iterations of the FastPD executable
% - lambda : parameter for  the rigitidy of the registration
% - label_factor : define te size of the label space at level n+1 depending on the size at level n
% - list of output :
% - registered_image : result of the image registration
% - time : time spent 
% - energy : final energy of the mrf

% an exemple of use is :
% [registered_image, time, energy, deformation_field] = mrf_framework('target_r.png', 'source_r.png', [11 11], 3, 5, 9, 10, 10, 0.5);


function [registered_image, time, energy, deformation_field] = mrf_framework(fixed_path, moving_path, node_grid_sizes, pyramidLevels, iterations, label_grid_size, num_max_iters, lambda, label_factor)
%     close all
%     clc;
    start = tic;
    fixed  = imread(fixed_path);
    moving = imread(moving_path);
    fixed  = rgb2gray(fixed);
    moving = rgb2gray(moving);
    
    figure('Name', 'target-source'); imshowpair(fixed, moving, 'diff');
    l = size(label_grid_size,2);
    m = size(num_max_iters,2);
    if(isinteger(power(l*m, 1/2)) == 1  )
        error('label_grid_sizes and num_max_iters must have the same length');
    end 
    if(label_factor <= 0)
        error('label_factor must be positive');
    end
    
    if(label_factor > 1)
        error('label_factor must be inferior or equal to 1');
    end
    
    dimensions = size(fixed);
    deformation_field = zeros(dimensions(1), dimensions(2));
    
    registered_image = moving;
    for j=1:iterations
        fprintf('pyramid level : %d, iterations : %d', 1, j);
       [registered_image, t, energy, Df] = mrf_registration(fixed, registered_image, node_grid_sizes, label_grid_size, num_max_iters, lambda);
       deformation_field = deformation_field + Df;
    end
%     disp(t);
%     disp(energy);
    if(pyramidLevels>1)
        for i=2:pyramidLevels
            label_grid_size = round(label_factor * label_grid_size);
            if rem(label_grid_size, 2) ~= 1
                label_grid_size = label_grid_size +1;
            end
            node_grid_sizes = 2*node_grid_sizes-1;
            for j=1:iterations
                fprintf('pyramid level : %d, iterations : %d', i, j);
                [registered_image, t, energy, Df] = mrf_registration(fixed, registered_image, node_grid_sizes, label_grid_size, num_max_iters, lambda);
                deformation_field = deformation_field + Df;
            end
%             disp(t);
%             disp(energy);
        end
    end
    time = toc(start);
    deformation_field = -1 * deformation_field;
    figure('Name', 'registered-target'); imshowpair(fixed, registered_image, 'diff');
    figure('Name', 'registered_image'); imshow(registered_image);
    
end 


%intermediate function that calls other functions to perform registration
% list of parameters !
% - fixed : matlab matrix containing the target image
% - moving : matlab matrix containing the source image
% - node_grid_sizes : list of 2 elements containing node grid sizes for each dimension, first element is height, second is width
% - label_grid_size : size of the label space
% - num_max_iters : number of maximum iterations of the FastPD executable
% - lambda : parameter for  the rigitidy of the registration
% list of output :
% - registered_image : result of the image registration
% - time : time spent 
% - energy : final energy of the mrf
function [registered_image, time, energy, Df] = mrf_registration(fixed, moving, node_grid_sizes, label_grid_sizes, num_max_iters, lambda)
    start = tic; 
    [dataset_path, uPotential, pPotential] = mrf_create_dataset(fixed, moving, node_grid_sizes, label_grid_sizes, num_max_iters, lambda);
    num_nodes = node_grid_sizes(1)*node_grid_sizes(2);

    num_labels = label_grid_sizes*label_grid_sizes;
    solution_path = mrf_solver(dataset_path, num_nodes, num_labels, num_max_iters);
    [registered_image, sol, Df] = mrf_warp(solution_path, moving, node_grid_sizes, label_grid_sizes);
    time = toc(start);
    energy = 0;
    % energy function needs to be corrected
    %energy = mrf_energy_calculate(uPotential, pPotential, sol, node_grid_sizes, label_grid_sizes);
end

% intermediate function that create a dataset usable in the FastPD executable
% list of parameters:
% - fixed : matlab matrix containing the target image
% - moving : matlab matrix containing the source image
% - node_grid_sizes : list of 2 elements containing node grid sizes for each dimension, first element is height, second is width
% - label_grid_size : size of the label space
% - num_max_iters : number of maximum iterations of the FastPD executable
% - lambda : parameter for  the rigitidy of the registration
% list of output :
% - dataset_path : path to where the input of the FastPD executable is saved
% - uPotential : matlab matrix containing unary potentials
% - pPotential : matlab matrix containing pairwise potentials
function [dataset_path, uPotential, pPotential] = mrf_create_dataset(fixed, moving, node_grid_sizes, label_grid_size, num_max_iters, lambda) 

   
    dimensions = size(moving);
    dim = size(dimensions, 2);

    if dim ~= size(node_grid_sizes,2)
        error ('Dimensions of images and node grid are not the same');
    end
    height = dimensions(1);
    width = dimensions(2);
    patch_size = 5*label_grid_size;
    if(rem(patch_size,2))~=1
        patch_size = patch_size+1;
    end
    margin = label_grid_size * patch_size;
    m_fixed = zeros( height + 2*margin, width + 2*margin);
    m_moving = m_fixed;
    
    m_fixed(margin+1:height+margin,margin+1:width+margin) = fixed+1;
    m_moving(margin+1:height+margin,margin+1:width+margin) = moving+1;

%     [N, D] = rat(height/width);   
%     r1 = node_grid_sizes(1)/node_grid_sizes(2);
%     r2 = (N+1)/(D+1);
%     if rem(r1, r2) ~=0
%         error('ratio error');
%     end
    
    % define number of nodes, labels, maximum iterations, and the size of the
    % patch 
    num_nodes = node_grid_sizes(1)*node_grid_sizes(2);
    num_nodes_y = node_grid_sizes(1) ;
    num_nodes_x = node_grid_sizes(2) ;
    
    if  rem(label_grid_size, 2) ~= 1
        error ('label_grid_size should be an odd number');
    end
    
    num_labels = label_grid_size*label_grid_size;    

     if(num_nodes ~= uint32(num_nodes))
         error('num_nodes must be integer')
     end
    if(num_nodes < 4)
        error('num_nodes must equal or superior to 4')
    end

    wNodeRange = floor(width/(num_nodes_x-1));
    hNodeRange = floor(height/(num_nodes_y-1));
    node_dist = wNodeRange;
    if wNodeRange ~= hNodeRange
        node_dist = round((wNodeRange+hNodeRange)/2);
%       the node grid must be a regular lattice i.e. horizontal and vertical spacing of nodes must be the same   
%       error('wNodeRange and hNodeRange are not equals');
    end 
    

    % calculate de number of edges according to the number of nodes
    %num_edges = 2*node_grid_sizes*(node_grid_sizes-1);
    num_edges =  (num_nodes_x-1)*num_nodes_y + num_nodes_x*(num_nodes_y-1)  ;

    % create list of edges
    edges = zeros(1, 2*num_edges);
    index = 1;
    for j=1:num_nodes_x
       for i=1:num_nodes_y
            if(i~=num_nodes_y)
              edges(index) =  num_nodes_x*(i-1)+j-1;
              edges(index+1) = num_nodes_x*(i-1)+j-1+num_nodes_x;
              index = index+2;
            end
            if(j~=num_nodes_x)
                edges(index) = num_nodes_x*(i-1)+j-1;
                edges(index+1) = num_nodes_x*(i-1)+j;
                index = index+2;
            end
        end
    end 

    %calculate unary potentials
    offset = floor(label_grid_size/2);
    uPotential = 255 * patch_size^2 * label_grid_size^2 * ones(num_labels, num_nodes);
    imnode = zeros(height+2*margin, width + 2*margin);
    rows = zeros(1, num_nodes_y*num_nodes_x);
    cols = rows;
    for i=0:num_nodes_x-1
        for j=0:num_nodes_y-1
    %       node n = i*(num_nodes_x-1) + j     
            if j==0
                h = 0:patch_size-1; 
                row = 1;
            else
                h = j*node_dist:j*node_dist+patch_size-1;
                row = j*node_dist+1;
            end
            
            if i==0
                w = 0:patch_size-1;
                col = 1;
            else
                w = i*node_dist:i*node_dist+patch_size-1;
                col = i*node_dist+1;
            end 
            h = h - offset;
            w = w - offset;
            for k=-floor(label_grid_size/2):floor(label_grid_size/2)
                for l=-floor(label_grid_size/2):floor(label_grid_size/2)
                    node_number = i*(num_nodes_x) + j +1;
                    label_number = (k+offset)*label_grid_size + l + offset + 1 ;
                    try
                        label_patch =  m_fixed(margin+h,margin+w) - m_moving(margin+h+k,margin+w+l);
                        potential = sum(label_patch.^2, dim);
                        potential = sum(potential);                   
                        uPotential(label_number, node_number) = potential;
                    catch ME
                        ME.stack.line
                        ME.message
                    end
                end
            end
            imnode(row+margin, col+margin) = 60;
            rows(node_number) = row;
            cols(node_number) = col;
            
        end
    end      
%     figure('Name', 'Imnode'); image(imnode); colormap hot
     pPotential = zeros(num_labels);
    
    % calculate pairwise potentials
    offset = ceil(num_labels/2);
    for j=1:num_labels
        for i=1:num_labels
            if(i~=j)
                b = rem(i-offset,label_grid_size);
                a = (i-offset-b)/label_grid_size;
                y = rem(j-offset,label_grid_size);
                x = (j-offset-y)/label_grid_size;
                d = sqrt( (b-y)^2 + (a-x)^2 );
                pPotential((j-1)*num_labels + i) = d/node_dist;
            end
        end
    end 
    
     % define list of weights
    wcosts = lambda * ones(1, num_edges);
    
    % write dataset in a binary file  
    dataset_path = sprintf("samples/dataset_%d_%d_%d.bin", num_nodes, num_labels, num_max_iters);
    output = fopen(dataset_path, 'wb');       
    type = 'uint32';
    fwrite(output, num_nodes, type);
    fwrite(output, num_edges, type);
    fwrite(output, num_labels, type);
    fwrite(output, num_max_iters, type);  
    fwrite(output, uPotential.', type);
    fwrite(output, edges, type);
    fwrite(output, pPotential, type);
    fwrite(output, wcosts, type);   
    fclose(output);
end

%intermediate function calling the FastPD executable
% list of parameters :
% - dataset_path : path to the dataset
% - num_nodes : number of nodes
% - num_labels : numbers of labels
% - num_max_iters : number of maximum iterations
% list of output :
% - solution_path : path to where should the ouput of the FastPD executabke should be saved 
function [solution_path] = mrf_solver(dataset_path, num_nodes, num_labels, num_max_iters)
        solution_path = sprintf('solutions/solution_%d_%d_%d.bin', num_nodes, num_labels, num_max_iters);
        command = sprintf('FastPD_manylabel.exe %s %s', dataset_path, solution_path);
        disp('Run FastPD_manylabel.exe...');
        system(command);
end

% intermediate function
% given the label set obatined using FastPD, it interpolates the full
% deformation field, and warp the source image.
% list of parameters :
% - solution_path : path of the output of the FastPD executable
% - moving : matlab matrix containing the source image
% - node_grid_sizes : list of 2 elements containing node grid sizes for each dimension, first element is height, second is width
% - label_grid_size : size of the label space
% list of output :
% - registered_image : result of the image registration
% - sol : matlab matrix containing reulsut of the FastPD executable
function [registered_image, sol, Df] = mrf_warp(solution_path, moving, node_grid_sizes, label_grid_size)
    file = fopen(solution_path, 'rb');
    solution = fread(file, [node_grid_sizes(1) node_grid_sizes(2)], 'uint32');
    sol = solution;
%     disp(solution);
    fclose(file);
    tmp = floor(label_grid_size/2);
    xd = rem(solution, label_grid_size);
    yd = (solution-xd)/label_grid_size;
    
    xd = xd - tmp;
    yd = -(yd - tmp);

    [height, width] = size(moving);
    num_nodes = node_grid_sizes(1)*node_grid_sizes(2);
    
    %write the deofmration field using interpolation
    valx= zeros(node_grid_sizes(1), node_grid_sizes(2));
    valy = valx;
    for i=1:num_nodes
            try
                valx(i) = xd(i);
                valy(i) = yd(i);
            catch
            end
    end
    
%     l = 0:label_grid_size*label_grid_size-1;
%     l = reshape(l, [label_grid_size label_grid_size]);
%     l = l';
%      disp(l);

    m = node_grid_sizes(1);
    n = node_grid_sizes(2);
    [X, Y] = meshgrid(0:n-1, 0:m-1);
    [Xq, Yq] = meshgrid(0:(n-1)/(width-1):n-1,0:(m-1)/(height-1):m-1);
    interpmethod = 'spline';
    ivalx = interp2(X,Y, valx, Xq, Yq, interpmethod);
    ivaly = interp2(X,Y, valy, Xq, Yq, interpmethod);
%     figure('Name', 'x-deformation field') ; surf(Xq,Yq,ivalx);
%     figure('Name', 'y-deformation field') ; surf(Xq,Yq,ivaly);
    Df = zeros(height, width, 2);
    Df(:,:,1) = -ivalx;
    Df(:,:,2) = ivaly;   
    registered_image = imwarp(moving, -Df, 'cubic', 'SmoothEdges', false);  
end


%intermediate function
% given the final label space for each node, calculate the total energy of
% the markov random field
% list of parameters :
% - uPotential : matlab matrix containing unary potentials
% - pPotential : matlab matrix containing pairwise potentials
% - solution : matlab matrix containing the result of the FastPD executable
% - node_grid_sizes : list of 2 elements containing node grid sizes for each dimension, first element is height, second is width
% - label_grid_size : size of the label space
% list of output :
% - energy : the total energy of the mrf
function [energy] = mrf_energy_calculate(uPotential, pPotential, solution, node_grid_sizes, label_grid_size)
    num_nodes = node_grid_sizes(1)*node_grid_sizes(2);
    num_labels = label_grid_size*label_grid_size;
    num_edges = (node_grid_sizes(1)-1)*node_grid_sizes(2) + node_grid_sizes(1)*( node_grid_sizes(2) - 1 );
    energy = 0; 
    solution = solution+1;
    
    for i=0:num_nodes-1
        energy = energy + uPotential(i*num_nodes + solution(i+1));
    end 
    
    edges = zeros(1, 2*num_edges);
    index = 1;
    
    for i=1:node_grid_sizes(1)
        for j=1:node_grid_sizes(2)
            if(i~=node_grid_sizes(1))
              edges(index) =  solution((j-1)*node_grid_sizes(1)+i);
              edges(index+1) = solution((j-1)*node_grid_sizes(1)+i+1);
              index = index+2;
            end
            if(j~=node_grid_sizes)
                edges(index) = solution((j-1)*node_grid_sizes(1)+i);
                edges(index+1) = solution((j)*node_grid_sizes(1)+i);
                index = index+2;
            end
        end
    end 
    
    for i=0:num_edges-1
        u1 = edges(2*i+1);
        u2 = edges(2*i+2);
        energy = energy + pPotential((u1-1)*num_labels + u2);
    end
           
end
    
    
