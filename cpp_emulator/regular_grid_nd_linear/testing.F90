subroutine test(array, size_x, size_y)
implicit none
integer,intent(inout) :: size_x, size_y
real*8,intent(inout) :: array(size_x, size_y)
integer :: i,j
do i=1,size_y
    do j=1,size_x
        array(j,i) = (i+j)
    enddo
enddo


end subroutine test