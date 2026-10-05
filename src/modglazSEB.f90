module modglazSEB
    implicit none

    private
    public :: calc_room_hin, calc_gap_h, gauss_solve, SEB_glaz

contains

    subroutine calc_room_hin(Ts_in, T_in, H, gamDeg, hin, Nu_in, RaH, Tmf, gamEff)
        real, intent(in)  :: Ts_in, T_in, H, gamDeg
        real, intent(out) :: hin, Nu_in, RaH, Tmf, gamEff

        real :: g, rho, cp, lam, mu, sinGam, RaCV
        real, parameter :: pi = acos(-1.0)

        if (H <= 0.0) then
            error stop 'H must be > 0.'
        end if

        g = 9.81
        rho = 1.225
        cp = 1005.0

        Tmf = T_in + 0.25 * (Ts_in - T_in)
        lam = 2.873e-3 + 7.76e-8 * Tmf
        mu = 3.723e-6 + 4.94e-8 * Tmf

        gamEff = gamDeg
        if (Ts_in > T_in) then
            gamEff = 180.0 - gamEff
        end if

        sinGam = sin(gamEff * pi / 180.0)
        RaH = (rho ** 2 * H ** 3 * g * cp * abs(Ts_in - T_in)) / (Tmf * mu * lam)

        if (gamEff >= 0.0 .and. gamEff < 15.0) then
            Nu_in = 0.13 * RaH ** (1.0 / 3.0)
        else if (gamEff >= 15.0 .and. gamEff <= 90.0) then
            ! ISO 15099: Ra_cv = 2.5e5 * (exp(0.72*gamma) / sin(gamma))**(1/5); both branches meet at Ra_H = Ra_cv
            RaCV = 2.5e5 * (exp(0.72 * gamEff) / max(sinGam, epsilon(1.0))) ** (1.0 / 5.0)
            if (RaH <= RaCV) then
                Nu_in = 0.56 * (RaH * sinGam) ** (1.0 / 4.0)
            else
                Nu_in = 0.13 * (RaH ** (1.0 / 3.0) - RaCV ** (1.0 / 3.0)) + &
                        0.56 * (RaCV * sinGam) ** (1.0 / 4.0)
            end if
        else if (gamEff > 90.0 .and. gamEff <= 179.0) then
            Nu_in = 0.56 * (RaH * sinGam) ** (1.0 / 4.0)
        else if (gamEff > 179.0 .and. gamEff <= 180.0) then
            Nu_in = 0.58 * RaH ** (1.0 / 5.0)
        else
            error stop 'gammaDeg must be in [0, 180].'
        end if

        hin = Nu_in * lam / H
    end subroutine calc_room_hin


    subroutine calc_gap_h(Ts, d_gas, rho_gas, mu_gas, c_gas, lam_gas, An, n_exp, h)
        real, intent(in) :: Ts(:), d_gas(:), rho_gas(:), mu_gas(:), c_gas(:), lam_gas(:)
        real, intent(in) :: An, n_exp
        real, allocatable, intent(out) :: h(:)

        integer :: ng
        real, allocatable :: TsL(:), TsR(:), Gr(:), Pr(:), Ra(:), Nu(:)

        ng = size(d_gas)
        allocate(h(ng))

        if (ng == 0) then
            return
        end if

        allocate(TsL(ng), TsR(ng), Gr(ng), Pr(ng), Ra(ng), Nu(ng))

        TsL = Ts(2:2 * ng:2)
        TsR = Ts(3:2 * ng + 1:2)

        Gr = 9.81 * d_gas ** 3 * abs(TsR - TsL) * rho_gas ** 2 / (283.0 * mu_gas ** 2)
        Pr = mu_gas * c_gas / lam_gas
        Ra = Gr * Pr
        Nu = An * Ra ** n_exp
        Nu = max(Nu, 1.0)

        h = Nu * (lam_gas / d_gas)
    end subroutine calc_gap_h


    subroutine gauss_solve(A_in, b_in, x)
        real, intent(in) :: A_in(:,:), b_in(:)
        real, allocatable, intent(out) :: x(:)

        real, allocatable :: A(:,:), b(:), temp_row(:)
        integer :: m, k, i, pivot_row
        real :: tol, pivot, pivot_abs, multiplier, temp_val

        if (size(A_in, 1) /= size(A_in, 2)) then
            error stop 'A must be square (m x m).'
        end if

        m = size(A_in, 1)
        if (size(b_in) /= m) then
            error stop 'size(b,1) must equal size(A,1).'
        end if

        allocate(A(m, m), b(m), x(m), temp_row(m))
        A = A_in
        b = b_in

        tol = epsilon(1.0) * max(1.0, maxval(sum(abs(A), dim=2))) * real(m)

        do k = 1, m - 1
            pivot_row = k
            pivot_abs = abs(A(k, k))

            do i = k + 1, m
                if (abs(A(i, k)) > pivot_abs) then
                    pivot_abs = abs(A(i, k))
                    pivot_row = i
                end if
            end do

            if (pivot_abs < tol) then
                error stop 'Matrix is singular or near-singular during elimination.'
            end if

            if (pivot_row /= k) then
                temp_row = A(k, :)
                A(k, :) = A(pivot_row, :)
                A(pivot_row, :) = temp_row

                temp_val = b(k)
                b(k) = b(pivot_row)
                b(pivot_row) = temp_val
            end if

            pivot = A(k, k)
            do i = k + 1, m
                if (abs(A(i, k)) < epsilon(1.0)) cycle
                multiplier = A(i, k) / pivot
                A(i, k:m) = A(i, k:m) - multiplier * A(k, k:m)
                b(i) = b(i) - multiplier * b(k)
                A(i, k) = 0.0
            end do
        end do

        if (abs(A(m, m)) < tol) then
            error stop 'Matrix is singular or near-singular at last pivot.'
        end if

        x = 0.0
        do i = m, 1, -1
            if (abs(A(i, i)) < tol) then
                error stop 'Zero/near-zero pivot encountered during back substitution.'
            end if
            if (i < m) then
                x(i) = (b(i) - sum(A(i, i + 1:m) * x(i + 1:m))) / A(i, i)
            else
                x(i) = b(i) / A(i, i)
            end if
        end do

        deallocate(temp_row)
    end subroutine gauss_solve


    subroutine SEB_glaz(sen_heat, lw_abs, sw_abs, T_in, &
                        emib, emif, lam_g, d_g,&
                        c_gas, rho_gas, mu_gas, lam_gas, d_gas,&
                        Ts_m, Ts)
        real, intent(in) :: sen_heat, lw_abs ! sensible heat flux and absorbed longwave radiation on the outer surface
        real, intent(in) :: T_in ! room air temperature, also the radiative temperature of the room
        real, intent(in) :: sw_abs(:) ! absorbed shortwave radiation on each glazing surface (2 per layer)
        real, intent(in) :: Ts_m(:), emib(:), emif(:), lam_g(:), d_g(:)
        real, intent(in) :: c_gas(:), rho_gas(:), mu_gas(:), lam_gas(:), d_gas(:)
        real, allocatable, intent(out) :: Ts(:)

        integer :: N, i, iter
        integer, parameter :: max_iter = 100 ! cap on the fixed-point iterations
        real, parameter :: sig = 5.67e-8
        real :: E_in, G_theta, window_h, An, n_exp_gap, h_in, res_max, res
        real :: Nu_in, RaH, Tmf, gamEff
        real, allocatable :: k(:), r(:), h_gap(:), Ts_star(:), Ts_old(:), A(:,:), B(:)

        N = size(lam_g)

        if (size(Ts_m) /= 2 * N) then
            error stop 'Ts_m must have length 2*N.'
        end if
        if (size(sw_abs) /= 2 * N) then
            error stop 'sw_abs must have length 2*N.'
        end if
        if (size(emib) /= N .or. size(emif) /= N .or. size(lam_g) /= N .or. size(d_g) /= N) then
            error stop 'emib, emif, lam_g, and d_g must have length N.'
        end if
        if (N > 1) then
            if (size(d_gas) /= N - 1 .or. size(rho_gas) /= N - 1 .or. size(mu_gas) /= N - 1 .or. &
                size(c_gas) /= N - 1 .or. size(lam_gas) /= N - 1) then
                error stop 'Gas property arrays must have length N-1.'
            end if
        end if

        allocate(Ts(2 * N), Ts_star(2 * N), Ts_old(2 * N), k(N), r(max(N - 1, 0)), A(2 * N, 2 * N), B(2 * N))
        Ts = Ts_m
        Ts_star = Ts_m

        E_in = sig * T_in ** 4
        G_theta = 90.0
        window_h = 3.0

        k = lam_g / d_g

        if (G_theta < 22.5) then
            G_theta = 0.0
        else if (G_theta < 67.5) then
            G_theta = 45.0
        else
            G_theta = 90.0
        end if

        if (G_theta < 22.5) then
            An = 0.16
            n_exp_gap = 0.28
        else if (G_theta < 67.5) then
            An = 0.1
            n_exp_gap = 0.31
        else
            An = 0.035
            n_exp_gap = 0.38
        end if

        if (N > 1) then
            do i = 1, N - 1
                r(i) = sig * emib(i) * emif(i + 1) / (1.0 - (1.0 - emib(i)) * (1.0 - emif(i + 1)))
            end do
        end if

        res_max = 0.01
        res = 1.0
        iter = 0

        do while (res > res_max .and. iter < max_iter)
            iter = iter + 1
            Ts_old = Ts

            call calc_room_hin(Ts(2 * N), T_in, window_h, G_theta, h_in, Nu_in, RaH, Tmf, gamEff)

            if (N > 1) then
                call calc_gap_h(Ts, d_gas, rho_gas, mu_gas, c_gas, lam_gas, An, n_exp_gap, h_gap)
            else
                allocate(h_gap(0))
            end if

            A = 0.0
            B = 0.0

            B(1) = lw_abs + sen_heat + sw_abs(1) ! lw_abs already includes the emissivity emif(1) (calclw)
            B(2 * N) = E_in * emib(N) + h_in * T_in + sw_abs(2 * N)

            A(1, 1) = sig * emif(1) * Ts(1) ** 3 + k(1)
            A(1, 2) = -k(1)
            A(2 * N, 2 * N - 1) = -k(N)
            A(2 * N, 2 * N) = sig * emib(N) * Ts(2 * N) ** 3 + k(N) + h_in

            if (N > 1) then
                do i = 1, N - 1
                    A(2 * i, 2 * i - 1) = -k(i)
                    A(2 * i, 2 * i) = k(i) + h_gap(i) + r(i) * Ts(2 * i) ** 3
                    A(2 * i, 2 * i + 1) = -(h_gap(i) + r(i) * Ts(2 * i + 1) ** 3)

                    A(2 * i + 1, 2 * i) = -(h_gap(i) + r(i) * Ts(2 * i) ** 3)
                    A(2 * i + 1, 2 * i + 1) = h_gap(i) + k(i + 1) + r(i) * Ts(2 * i + 1) ** 3
                    A(2 * i + 1, 2 * i + 2) = -k(i + 1)

                    B(2 * i) = sw_abs(2 * i)
                    B(2 * i + 1) = sw_abs(2 * i + 1)
                end do
            end if

            call gauss_solve(A, B, Ts_star)

            Ts = Ts + 0.5 * (Ts_star - Ts)
            res = maxval(abs(Ts - Ts_old))

            if (allocated(h_gap)) then
                deallocate(h_gap)
            end if
        end do

        if (res > res_max) then
            write(0, *) 'WARNING: SEB_glaz did not converge in ', max_iter, ' iterations, residual ', res, ' K'
        end if
    end subroutine SEB_glaz

end module modglazSEB