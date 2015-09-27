/*
 * Copyright 2010 Tim Perkins. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY TIM PERKINS ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of Tim Perkins.
 */

/*
 * pymex.cpp
 *
 * Run Python in Matlab. A matlab module is also provided for printing and
 * exporting data to matlab. mex_print(...) works like python's built-in
 * print function. It accepts a variable number of inputs. export(...) uses
 * matlab's assignin function to put variables in the caller scope. It accepts
 * a variable number of input pairs, the new variable name in matlab and
 * the variable. Only numbers (int, long, float, etc) are supported for now.
 */

/*
Compile with:

g++ pymex.cpp -o pymex.mexglx -I/opt/matlab/2009A/extern/include/       \
-Wl,-rpath-link,/opt/matlab/2009A/bin/glnx86 -L/opt/matlab/2009A/bin/glnx86 \
-lmx -lmex -lmat -lm -I/usr/include/python2.6 -I/usr/include/python2.6  \
-L/usr/lib/python2.6/config -lpthread -ldl -lutil -lm -lpython2.6 -rdynamic -shared
*/

#include <dlfcn.h>
#include <mex.h>
#include <matrix.h>
#include <string.h>
#include <python2.7/Python.h>

#include <string>
#include <vector>

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)
#ifdef LIBPYTHON
#define LIBPYTHONSO QUOTEME(LIBPYTHON)
#else
#define LIBPYTHONSO "/usr/lib/libpython2.7.so"
#endif

static PyThreadState * thread_save;
static PyObject * matlab_module;
static PyObject * matlab_error;
static PyObject * matlab_push(PyObject * self, PyObject * args);
static PyObject * matlab_pull(PyObject * self, PyObject * args);
static PyObject * matlab_mex_print(PyObject * self, PyObject * args);
static void
recursive_matlab_fill(PyObject * object,
                      double * const data,
                      const mwSize num_dims,
                      const mwSize * const dims);
static bool float_check(PyObject * object);
static bool string_check(PyObject * object);
static void nudge_object(PyObject * object);
static void guess_dims(PyObject * tuple, mwSize & num_dims, mwSize * & dims);
static bool verify_matrix(PyObject * tuple, const mwSize num_dims, const mwSize * const dims);
template <class T> static PyObject *
recursive_matlab_pack(const T * const data,
                      const char * const format_type,
                      const mwSize num_dims,
                      const mwSize * const dims);

static PyMethodDef matlab_methods[] = {
  { "push", matlab_push, METH_VARARGS, "" },
  { "pull", matlab_pull, METH_VARARGS, "" },
  { "mex_print", matlab_mex_print, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static PyObject *
mat2py_convert_float(double value)
{
  return PyFloat_FromDouble(value);
}

static PyObject *
mat2py_convert_unsignedlonglong(unsigned long long value)
{
  // these both make new references
  PyObject * pything = PyLong_FromUnsignedLongLong(value);
  PyObject * try_int = PyNumber_Int(pything);
  if (try_int == NULL) {
    return pything;
  } else {
    Py_XDECREF(pything);
    return try_int;
  }
}

static PyObject *
mat2py_convert_longlong(long long value)
{
  // these both make new references
  PyObject * pything = PyLong_FromLongLong(value);
  PyObject * try_int = PyNumber_Int(pything);
  if (try_int == NULL) {
    return pything;
  } else {
    Py_XDECREF(pything);
    return try_int;
  }
}

template <typename T>
static T
mat2py_numeric_accessor(const mxArray * const arr, const size_t i)
{
  return ((T *) mxGetData(arr))[i];
}

static const mxArray * const
mat2py_cell_accessor(const mxArray * const cell, const size_t i)
{
  return mxGetCell(cell, i);
}

/**
 * Matlab has N dimensional arrays. Python doesn't have arrays, it just has nestled sequences. This
 * function turns arrays, or cells, or whatever, into nestled sequences.
 *
 * @param converter_func The function to convert mxArrays to PyObjects.
 * @param accessor_func The function to access an element of a mxArray.
 * @param level The dimension which is acting as the top level. Level = 0 inidcated the highest
 * level. Level = (num_dims - 2) indicates the lowest level, a N by M matrix.
 * @param offset The offset is the index of the first element of the related data subset.
*/
template <typename CONVERTER_FUNC, typename ACCESSOR_FUNC>
static PyObject *
mat2py_helper2(const mxArray * const matthing,
               CONVERTER_FUNC converter_func,
               ACCESSOR_FUNC accessor_func,
               const size_t level = 0,
               const size_t offset = 0,
               const bool check_empty = true)
{
  PyObject * pything;

  // get dimensions
  mwSize num_dims = mxGetNumberOfDimensions(matthing);
  mwSize * dims = (mwSize *) mxGetDimensions(matthing);

  // matlab can have 0 dimension, so we should check this at least once
  if (check_empty) {
    for (size_t i = 0; i < num_dims; i++) {
      if (dims[i] == 0) {
        Py_INCREF(Py_None);
        return Py_None;
      }
    }
  }

  if (num_dims - level == 2) {
    // this is a N by M
    if (dims[0] == 1 && dims[1] == 1) {
      // don't make a sequence, this is just 1 element
      // this should make a new reference
      pything = converter_func(accessor_func(matthing, offset));
    } else if (dims[0] == 1 || dims[1] == 1) {
      // make just one sequence, this is a 1 by N
      size_t len = dims[0] * dims[1];
      // new reference
      pything = PyTuple_New(len);
      for (size_t i = 0; i < len; i++) {
        // new reference
        PyObject * item = converter_func(accessor_func(matthing, offset + i));
        // TODO check for NULL?
        // steals reference
        PyTuple_SetItem(pything, i, item);
      }
    } else {
      // this is a real N by M, new reference
      pything = PyTuple_New(dims[0]);
      for (size_t i = 0; i < dims[0]; i++) {
        // build a sub tuple, new reference
        PyObject * subtuple = PyTuple_New(dims[1]);
        for (size_t j = 0; j < dims[1]; j++) {
          // new reference, remember that matlab is colomn major
          PyObject * item = converter_func(accessor_func(matthing, offset + i + dims[0] * j));
          // TODO check for NULL?
          // steals reference
          PyTuple_SetItem(subtuple, j, item);
        }
        // add it to the big tuple, steals reference
        PyTuple_SetItem(pything, i, subtuple);
      }
    }
  } else if (num_dims - level > 2) {
    // the new offset is the product of the lower dimensions
    size_t new_offset = 1;
    for (size_t z = 0; z < num_dims - level - 1; z++) {
      new_offset *= dims[z];
    }
    // bigger then a 2 by 2
    size_t len = dims[num_dims - level - 1];
    pything = PyTuple_New(len);
    for (size_t i = 0; i < len; i++) {
      // recurse using the new offset, returns new reference
      PyObject * item = mat2py_helper2(matthing, converter_func, accessor_func, level + 1,
                                       new_offset * i + offset, false);
      // steals reference
      PyTuple_SetItem(pything, i, item);
    }
  } else {
    // this shouldn't be possible
    PyErr_SetString(matlab_error, "no dimension, invalid mxArray");
    return NULL;
  }

  return pything;
}

static PyObject *
mat2py(const mxArray * const matthing)
{
  // make a pyobject
  PyObject * pything;

  // check for null, will happen for unpopulated cells
  if (matthing == NULL) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  if (mxIsNumeric(matthing)) {
    // get details...
    mwSize num_dims = mxGetNumberOfDimensions(matthing);
    mwSize * dims = (mwSize *) mxGetDimensions(matthing);
    // this is numeric, but what kind?
    // mat2py_helper2 returns new reference
    if (mxIsDouble(matthing)) {
      double * data = (double *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_float,
                               mat2py_numeric_accessor<double>);
    } else if (mxIsSingle(matthing)) {
      float * data = (float *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_float,
                               mat2py_numeric_accessor<float>);
    } else if (mxIsInt64(matthing)) {
      int64_t * data = (int64_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_longlong,
                               mat2py_numeric_accessor<int64_t>);
    } else if (mxIsUint64(matthing)) {
      uint64_t * data = (uint64_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_unsignedlonglong,
                               mat2py_numeric_accessor<uint64_t>);
    } else if (mxIsInt32(matthing)) {
      int32_t * data = (int32_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_longlong,
                               mat2py_numeric_accessor<int32_t>);
    } else if (mxIsUint32(matthing)) {
      uint32_t * data = (uint32_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_longlong,
                               mat2py_numeric_accessor<uint32_t>);
    } else if (mxIsInt16(matthing)) {
      int16_t * data = (int16_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_longlong,
                               mat2py_numeric_accessor<int16_t>);
    } else if (mxIsUint16(matthing)) {
      uint16_t * data = (uint16_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_longlong,
                               mat2py_numeric_accessor<uint16_t>);
    } else if (mxIsInt8(matthing)) {
      int8_t * data = (int8_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_longlong,
                               mat2py_numeric_accessor<int8_t>);
    } else if (mxIsUint8(matthing)) {
      uint8_t * data = (uint8_t *) mxGetData(matthing);
      pything = mat2py_helper2(matthing, mat2py_convert_longlong,
                               mat2py_numeric_accessor<uint8_t>);
    }
  } else if (mxIsChar(matthing)) {
    // flat strings only please
    if (mxGetNumberOfDimensions(matthing) != 2 ||
        (mxGetDimensions(matthing)[0] != 1 && mxGetDimensions(matthing)[1] != 1)) {
      PyErr_SetString(matlab_error, "only flat strings are allowed");
      return NULL;
    }
    // put the matlab string in the c++ string
    std::string str;
    str.resize(mxGetNumberOfElements(matthing), 'x');
    if (mxGetString(matthing, &str[0], str.size() + 1)) {
      PyErr_SetString(matlab_error, "could not read string");
      return NULL;
    }
    // new reference
    pything = PyString_FromString(str.c_str());
  } else if (mxIsCell(matthing)) {
    // use this function as the conveter function, since cells can contain anything
    // returns a new reference
    pything = mat2py_helper2(matthing, mat2py, mat2py_cell_accessor);
  } else if (mxIsStruct(matthing)) {
    PyErr_SetString(matlab_error, "not implemented");
    return NULL;
  } else {
    // not supported
    PyErr_SetString(matlab_error, "this type is not supported");
    return NULL;
  }

  // return the new reference
  return pything;
}

static PyObject *
matlab_pull(PyObject * self, PyObject * args)
{
  // get the list, new reference
  PyObject * fast_args = PySequence_Fast(args, "matlab_import fast args");
  size_t size_args = PySequence_Fast_GET_SIZE(fast_args);

  // check for no input
  if (size_args == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  // the args, set scope to 'caller'
  mxArray * mexargs[2];
  mexargs[0] = mxCreateString("caller");

  // make the super tuple
  PyObject * new_super_tuple = PyTuple_New(size_args);

  for (size_t i = 0; i < size_args; i++) {
    // get the item, borrowed reference
    PyObject * object_name = PySequence_Fast_GET_ITEM(fast_args, i);
    if (object_name == NULL || !PyString_CheckExact(object_name)) {
      PyErr_SetString(PyExc_TypeError, "variable names must be strings");
      return NULL;
    }

    // get the variable name as an arg
    mexargs[1] = mxCreateString(PyString_AsString(object_name));

    // do the call, we have to catch exceptions or else matlab will throw
    int return_value;
    mxArray * matthing;
    try {
      return_value = mexCallMATLAB(1, &matthing, 2, mexargs, "evalin");
    } catch (...) {
      return_value = -1;
    }
    if (return_value != 0) {
      PyErr_SetString(matlab_error, "evalin has failed");
      return NULL;
    }

    // convert the value and stuff it, new reference
    PyObject * pything = mat2py(matthing);
    // steals reference
    PyTuple_SetItem(new_super_tuple, i, pything);
  }

  // if there is only one thing, just return that
  if (size_args == 1) {
    return PyTuple_GetItem(new_super_tuple, 0);
  }

  return new_super_tuple;
}

static void
recursive_matlab_fill(PyObject * object,
                      double * const data,
                      const mwSize num_dims,
                      const mwSize * const dims)
{
  // check for no dims
  if (num_dims < 1) {
    return;
  }

  if (num_dims == 2) {
    // fill the terminal values
    for (size_t i = 0; i < dims[0]; i++) {
      PyObject * row = PySequence_GetItem(object, i);
      for (size_t j = 0; j < dims[1]; j++) {
        size_t index = i + dims[0] * j;
        data[index] = PyFloat_AsDouble(PySequence_GetItem(row, j));
      }
    }
  } else {
    // how many elements do we skip ahead?
    size_t elements_per = 1;
    for (size_t i = 0; i < num_dims - 1; i++) {
      elements_per *= dims[i]; // sum all lower dims
    }

    // fill more data
    for (size_t i = 0; i < dims[num_dims - 1]; i++) {
      // get the new tuple
      PyObject * subtuple = PySequence_GetItem(object, i);
      // what is the new matlab data? shift to the data we want
      double * const new_data = data + elements_per * i;
      // recurse
      recursive_matlab_fill(subtuple, new_data, num_dims - 1, dims);
    }
  }
}

static bool
float_check(PyObject * object)
{
  // check if we support the object
  if (PyFloat_Check(object)) return true;

  PyObject * float_check;
  float_check = NULL;
  if (PyObject_HasAttrString(object, "__float__")) {
    float_check = PyObject_GetAttrString(object, "__float__");
  }
  return PyCallable_Check(float_check);
}

static bool
string_check(PyObject * object)
{
  // check if we support the object
  if (PyString_Check(object)) return true;

  PyObject * string_check;
  string_check = NULL;
  if (PyObject_HasAttrString(object, "__str__")) {
    string_check = PyObject_GetAttrString(object, "__str__");
  }
  return PyCallable_Check(string_check);
}

static void
guess_dims(PyObject * tuple, mwSize & num_dims, mwSize * & dims)
{
  Py_ssize_t length;
  PyObject * next;
  mwSize * new_dims, i;

  // initialize variable for the loop
  num_dims = 0;
  dims = NULL;
  next = tuple;

  // loop through the first element of each subsequence
  while (PySequence_Check(next)) {
    // get the length/dimension
    length = PySequence_Length(next);

    // add to dims
    new_dims = new mwSize[num_dims + 1];
    for (i = 0; i < num_dims; i++) {
      new_dims[i + 1] = dims[i];
    }
    delete [] dims;
    new_dims[0] = (mwSize) length;
    num_dims++;
    dims = new_dims;

    // check the next one
    next = PySequence_GetItem(next, 0);
  }

  // swap the last two
  mwSize tmp = dims[1];
  dims[1] = dims[0];
  dims[0] = tmp;

  return;
}

static bool
verify_matrix(PyObject * object, const mwSize num_dims, const mwSize * const dims)
{
  bool result;
  PyObject * element;

  // test if this is a tuple
  if (!PySequence_Check(object)) {
    return false;
  }

  // do we have a sequence of sequences?
  if (num_dims == 2) {
    if (PySequence_Length(object) != dims[0]) {
      return false;
    }
    // this is a sequence of numbers, check the type
    for (size_t i = 0; i < dims[0]; i++) {
      PyObject * subtuple = PySequence_GetItem(object, i);
      if (PySequence_Length(subtuple) != dims[1]) {
        return false;
      }
      for (size_t j = 0; j < dims[1]; j++) {
        PyObject * element = PySequence_GetItem(subtuple, j);
        if (!float_check(element)) {
          return false;
        }
      }
    }
    return true;
  } else {
    if (PySequence_Length(object) != dims[num_dims - 1]) {
      return false;
    }
    // this is a sequence of sequences, check all sub sequences
    result = true;
    for (size_t i = 0; i < dims[num_dims - 1]; i++) {
      element = PySequence_GetItem(object, i);
      result = result && verify_matrix(element, num_dims - 1, dims);
    }
    return result;
  }
}

static PyObject *
matlab_push(PyObject * self, PyObject * args)
{
  const int nlhs = 0, nrhs = 3;
  mxArray * prhs[3];
  bool error;
  int return_value;
  char * variable_name;
  double * variable;
  Py_ssize_t size_args, i, j;
  PyObject * fast_args;
  PyObject * object_name, * object;
  mwSize * all_num_dims;
  mwSize ** all_dims;

  // get the list
  fast_args = PySequence_Fast(args, "matlab_export fast args");
  size_args = PySequence_Fast_GET_SIZE(fast_args);

  // check that the array size is a multiple of 2
  if (size_args % 2 != 0) {
    PyErr_SetString(PyExc_KeyError, "arguments must be in pairs");
    return NULL;
  }

  // create the arrays to hold the dimensions
  all_num_dims = new mwSize[size_args / 2];
  all_dims = new mwSize *[size_args / 2];
  for (i = 0; i < size_args / 2; i++) {
    all_dims[i] = NULL;
  }

  // start with no error
  error = false;

  // because we don't want to return from this function before deleting
  // all the variables, we use a try catch block
  try {
    // this for loop is a little weird, j is incremented
    // and i is always twice j, this is because we read in variables
    // two at a time (name and value)
    for (i = j = 0; i < size_args; i = 2 * ++j) {
      // get the item
      object_name = PySequence_Fast_GET_ITEM(fast_args, i);
      object = PySequence_Fast_GET_ITEM(fast_args, i + 1);

      // we should have a string we can use for a variable name
      if (!PyString_CheckExact(object_name)) {
        PyErr_SetString(PyExc_TypeError, "variable names must be strings");
        throw 0;
      }

      // TODO unhack this...
      if (PySequence_Check(object) && !PyString_Check(object)) {
        // verify the matrix, allocates dims with new []
        guess_dims(object, all_num_dims[j], all_dims[j]);
        // for (int x = 0; x < all_num_dims[j]; x++) {
        //   mexPrintf("-> %d\n", all_dims[j][x]);
        // }
        if (!verify_matrix(object, all_num_dims[j], all_dims[j])) {
          PyErr_SetString(PyExc_TypeError, "variables must be reformable to matrices");
          throw 0;
        }
      } else if (PyNumber_Check(object)) {
        // do nothing
      } else if (PyString_Check(object)) {
        // do nothing
      } else {
        PyErr_SetString(PyExc_TypeError, "type not supported");
        throw 0;
      }
    }

    // set to 'caller'
    prhs[0] = mxCreateString("caller");

    // this for loop is a little weird, j is incremented
    // and i is always twice j, this is because we read in variables
    // two at a time (name and value)
    for (i = j = 0; i < size_args; i = 2 * ++j) {
      // get the item
      object_name = PySequence_Fast_GET_ITEM(fast_args, i);
      object = PySequence_Fast_GET_ITEM(fast_args, i + 1);

      // get the variable name
      variable_name = PyString_AsString(object_name);
      prhs[1] = mxCreateString(variable_name);

      if (PySequence_Check(object) && !PyString_Check(object)) {
        // get the value
        prhs[2] = mxCreateNumericArray(all_num_dims[j], all_dims[j], mxDOUBLE_CLASS, mxREAL);
        variable = (double *) mxGetData(prhs[2]);
        recursive_matlab_fill(object, variable, all_num_dims[j], all_dims[j]);
      } else if (PyNumber_Check(object)) {
        // make a scalar
        prhs[2] = mxCreateDoubleScalar(PyFloat_AsDouble(object));
      } else if (PyString_Check(object)) {
        // make a string
        prhs[2] = mxCreateString(PyString_AsString(object));
      }

      // did errors happen?
      if (PyErr_Occurred() != NULL) {
        throw 0;
      }

      // do the call
      try {
        return_value = mexCallMATLAB(nlhs, NULL, nrhs, prhs, "assignin");
      } catch (...) {
        return_value = 1;
      }

      if (return_value != 0) {
        // we should raise an exception here
        PyErr_SetString(matlab_error, "assignin has failed");
        throw 0;
      }
    }
  } catch (...) {
    error = true;
  }

  // delete everything we allocated
  delete [] all_num_dims;
  for (i = 0; i < size_args / 2; i++) {
    delete [] all_dims[i];
  }
  delete [] all_dims;

  // return the value finally
  if (error) {
    return NULL;
  } else {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject *
matlab_mex_print(PyObject * self, PyObject * args)
{
  char * string;
  Py_ssize_t size_args, i;
  PyObject * fast_args;
  PyObject * object, * str_check;

  // get the list
  fast_args = PySequence_Fast(args, "matlab_print fast args");
  size_args = PySequence_Fast_GET_SIZE(fast_args);

  if (size_args == 0) {
    // do nothing
    Py_INCREF(Py_None);
    return Py_None;
  }

  // loop through all input
  for (i = 0; i < size_args; i++) {
    // get the object
    object = PySequence_Fast_GET_ITEM(fast_args, i);

    // check if we support the object and convert it
    if (PyObject_HasAttrString(object, "__str__")) {
      str_check = PyObject_GetAttrString(object, "__str__");
    } else {
      str_check = NULL;
    }
    if (object == NULL || !(PyString_Check(object) || PyCallable_Check(str_check))) {
      PyErr_SetString(PyExc_TypeError, "variables must be strings (or have __str__)");
      return NULL;
    }
  }

  for (i = 0; i < size_args; i++) {
    // get the object
    object = PySequence_Fast_GET_ITEM(fast_args, i);

    // get the string
    if (!PyString_Check(object)) {
      object = PyObject_Str(object);
    }
    string = PyString_AsString(object);

    // did errors happen?
    if (PyErr_Occurred() != NULL) {
      return NULL;
    }

    // print a space?
    if (i > 0) {
      mexPrintf(" ");
    }

    // send that string to matlab
    mexPrintf(string);
  }

  // finish with a new line
  mexPrintf("\n");

  Py_INCREF(Py_None);
  return Py_None;
}

void
mexExit(void)
{
  if (Py_IsInitialized()) {
    PyEval_RestoreThread(thread_save);
    Py_Finalize();
  }
}

void
mexFunction(int nlhs, mxArray * plhs[ ], int nrhs, const mxArray * prhs[ ])
{
  // register mexExit
  mexAtExit(mexExit);

  // make sure one input
  if (nrhs != 1) {
    mexErrMsgIdAndTxt("pymex:args", "exactly 1 input allowed\n");
  }

  // make sure no output
  if (nlhs != 0 && nlhs != 1) {
    mexErrMsgIdAndTxt("pymex:args", "0 or 1 output allowed\n");
  }

  // fill this with python code
  std::string command_string;

  // test if input is cell of strings
  if (mxIsCell(prhs[0])) {
    if (mxGetNumberOfDimensions(prhs[0]) == 2 &&
        (mxGetDimensions(prhs[0])[0] == 1 || mxGetDimensions(prhs[0])[1] == 1)) {
      // make sure each element is a string
      size_t numel = mxGetNumberOfElements(prhs[0]);
      for (size_t i = 0; i < numel; i++) {
        mxArray * cell = mxGetCell(prhs[0], i);
        if (mxIsChar(cell)) {
          std::string piece;
          piece.resize(mxGetNumberOfElements(cell), 'x');
          if (mxGetString(cell, &piece[0], piece.size() + 1)) {
            mexErrMsgIdAndTxt("pymex:args", "could not read string\n");
          }
          command_string.append(piece);
          command_string.append("\n");
        } else {
          mexErrMsgIdAndTxt("pymex:args", "cell array had a non-string\n");
        }
      }
    } else {
      mexErrMsgIdAndTxt("pymex:args", "cell array of strings should be 1 by N\n");
    }
  } else if (mxIsChar(prhs[0])) {
    command_string.resize(mxGetNumberOfElements(prhs[0]), 'x');
    if (mxGetString(prhs[0], &command_string[0], command_string.size() + 1)) {
      mexErrMsgIdAndTxt("pymex:args", "could not read string\n");
    }
  } else {
    mexErrMsgIdAndTxt("pymex:args", "command must be a string or a cell of strings\n");
  }

  // This is a hack, because Matlab and Python don't get along, and I can't
  // figure out why exactly, why aren't Python's symbols normally global?
  if (!dlopen(LIBPYTHONSO, RTLD_LAZY | RTLD_GLOBAL)) {
    mexErrMsgIdAndTxt("pymex:lib", "cannot load libpython as global\n%s\n", dlerror());
  }

  if (!Py_IsInitialized()) {
    // start python
    Py_Initialize();

    // initialize the special matlab module
    matlab_module = Py_InitModule3("matlab", matlab_methods, "");

    // add the matlab error exception, new reference
    matlab_error = PyErr_NewException((char *) "matlab.error", NULL, NULL);
    // steals reference
    PyModule_AddObject(matlab_module, "error", matlab_error);
  } else {
    // get the GIL
    PyEval_RestoreThread(thread_save);
  }

  // parse the python string
  int return_value = PyRun_SimpleString(command_string.c_str());

  // return the status if needed
  if (nrhs == 1) {
    plhs[0] = mxCreateLogicalScalar(!return_value);
  }

  // release GIL while we go back to matlab
  thread_save = PyEval_SaveThread();
}
