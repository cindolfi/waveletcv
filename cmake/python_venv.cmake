include(FindPython)

if (Python_FOUND)
    set(VENV_PATH "${CMAKE_BINARY_DIR}/venv")
    message(CHECK_START "Creating python virtual environment")
    message(VERBOSE "VENV_PATH ${VENV_PATH}")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m venv ${VENV_PATH}
        RESULT_VARIABLE VENV_RESULT
    )
    if (VENV_RESULT GREATER 0)
        message(CHECK_FAIL "Failed")
        set(VENV_CREATED False)
    else()
        message(CHECK_PASS "Created")
        set(VENV_CREATED True)
        set(VENV_PYTHON_EXECUTABLE "${VENV_PATH}/bin/python")
        set(VENV_PIP_EXECUTABLE "${VENV_PATH}/bin/pip")
    endif()
endif()

function(pip_install package)
    message(STATUS "Installing python package ${package}")
    execute_process(
        COMMAND ${VENV_PIP_EXECUTABLE} install ${package}
        RESULT_VARIABLE installed
    )
    if (installed GREATER 0)
        message(FATAL_ERROR "Failed to install ${package}")
    endif()
endfunction()

function(pip_install_requirements requirements_file)
    message(STATUS "Installing python requirements: ${requirements_file}")
    execute_process(
        COMMAND ${VENV_PIP_EXECUTABLE} install -r "${requirements_file}"
        RESULT_VARIABLE requirements_installed
    )
    if (requirements_installed GREATER 0)
        message(FATAL_ERROR "Failed to install requirements: ${requirements_file}")
    endif()
endfunction(pip_install_requirements)

function(execute_python)
    execute_process(
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${ARGV}
    )
endfunction(execute_python)
