# Overlay test suites — per-target consistency / fuzz / bench /
# perf drivers for the hand-written kernels.

find_program(FYPP fypp REQUIRED)

# Helper for per-target wrapper shims:
#   add_fypp_source(<out_var> <shim.fypp> <body_dir>)
# generates <build>/.../<basename>.f90 from <shim.fypp>, with -I
# pointing at <body_dir> so #:include resolves the shared body file.
function(add_fypp_source out_var shim_path body_dir)
    get_filename_component(_stem ${shim_path} NAME_WE)
    set(_out ${CMAKE_CURRENT_BINARY_DIR}/${_stem}.f90)
    # Depend on every .fypp in body_dir so edits to the shared body
    # (or any future #:include sibling) trigger regeneration.
    file(GLOB _body_fypps CONFIGURE_DEPENDS ${body_dir}/*.fypp)
    add_custom_command(
        OUTPUT ${_out}
        COMMAND ${FYPP} -I ${body_dir} ${shim_path} ${_out}
        DEPENDS ${shim_path} ${_body_fypps}
        COMMENT "fypp ${shim_path} -> ${_out}"
        VERBATIM
    )
    set(${out_var} ${_out} PARENT_SCOPE)
endfunction()

add_subdirectory(epblas-parallel)
add_subdirectory(epblas-openblas)
