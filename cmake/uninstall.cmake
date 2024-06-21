
set(INSTALL_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt")

if(NOT EXISTS ${INSTALL_MANIFEST})
    message(FATAL_ERROR "Cannot find install manifest: '${INSTALL_MANIFEST}'")
endif()

file(STRINGS ${INSTALL_MANIFEST} installed_files)
foreach(file ${installed_files})
    if(EXISTS ${file})
        message(STATUS "Removing file: '${file}'")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E remove "${file}"
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
            RESULT_VARIABLE failed
        )
        if(failed)
            message(FATAL_ERROR "Failed to remove file: '${file}'.")
        endif()
    else()
        MESSAGE(VERBOSE "File '${file}' does not exist.")
    endif()
endforeach(file)
