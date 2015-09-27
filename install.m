function install(toolbox_dir, python_lib, python_dir, compiler)
% INSTALL(TOOLBOX_DIR, PYTHON_LIB, PYTHON_DIR, COMPILER)
%
% Compile and install pymex.
%
% toolbox_dir is Matlab's toolbox directory. It defaults to
% '/usr/local/matlab/toolbox/'.
%
% python_lib is the location of the Python shared library. It defaults to
% '/usr/lib/x86_64-linux-gnu/libpython2.7.so'.
%
% python_dir is the directory containing the Python headers. It defaults to
% '/usr/include/python2.7/'.
%
% compiler is the C++ compiler to use, typically 'g++', although you may want to
% use 'g++-4.7' to ensure compatibility.

    if ~exist('toolbox_dir', 'var') || isempty(toolbox_dir)
        toolbox_dir = '/usr/local/matlab/toolbox/';
    else
        if toolbox_dir(1) ~= '/'
            error('The toolbox directory must be given as an absolute path.');
        end
    end
    install_dir = [toolbox_dir, '/pymex/'];

    if ~exist('python_lib', 'var') || isempty(python_lib)
        python_lib = '/usr/lib/x86_64-linux-gnu/libpython2.7.so';
    end

    if ~exist('python_dir', 'var') || isempty(python_dir)
        python_dir = '/usr/include/python2.7/';
    end

    compiler_extra = '';
    if exist('compiler', 'var') && isempty(python_lib)
        % TODO still uses the wrong libraries, still gives warning
        compiler_extra = ['CXX="', compiler, '" '];
    end

    mex_command = ['mex -v ', compiler_extra, 'pymex.cpp ', ...
                   '-I', python_dir, ' -ldl ', python_lib, ' ', ...
                   '-DLIBPYTHON=', python_lib];

    disp(' ');
    disp(' ');
    disp('Now installing pymex. It will be installed here:');
    disp(' ');
    disp(['    ', install_dir]);
    disp(' ');
    disp('When building, mex needs to be able to link against');
    disp('libpythonX.Y and libdl. You may need to adjust the mex');
    disp('command to get everything working. Pymex also needs to ');
    disp('load libpythonX.Y during runtime, it will look here: ');
    disp(' ');
    disp(['    ', python_lib]);
    disp(' ');
    disp('...or else fail. The location of the library can be ');
    disp('adjusted by changing the ''python_lib'' argument. Type');
    disp('''help install.m'' to see usage.');
    disp(' ');
    disp(' ');
    disp(' < Press any key to continue >');
    disp(' ');
    disp(' ');
    pause;

    disp('Now checking for the toolbox directory...');
    disp(['(Looking for ', toolbox_dir, ')']);
    if ~exist(toolbox_dir, 'dir')
        error('Cannot find the toolbox directory.');
    end
    disp('Now checking for the Python library...');
    if ~exist(python_lib, 'file')
        error([python_lib, ' does not exist.']);
    end
    disp('Now compiling pymex.cpp with command:');
    disp(' ');
    disp(['     ', mex_command]);
    disp(' ');
    eval(mex_command);
    disp('Now installing pymex...');
    if ~movefile('./pymex.mex*', install_dir, 'f')
        error('Cannot install pymex files. Do you need to be root?');
    end
    disp('Now updating the path...');
    path(path, install_dir);
    if savepath ~= 0
        error('Cannot update path.');
    end
end
