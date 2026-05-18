module greeter_mod
    implicit none
contains
    subroutine greet(name)
        character(len=*), intent(in) :: name
        call format_name(name)
    end subroutine greet

    subroutine format_name(n)
        character(len=*), intent(in) :: n
        print *, n
    end subroutine format_name
end module greeter_mod
