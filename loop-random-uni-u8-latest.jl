# using BilevelJuMP
# using Ipopt
using JuMP
using Test
using HiGHS
# using MibS_jll
using Atom
using Random, Distributions
using DelimitedFiles
using Statistics

function f(x)
    summ=0
    for i in 1:length(x)
        summ += x[i]^2
    end
    return sqrt(summ)
end

n = 73 #73
m = 189 #189
u1 = 3
u2 = 5

# You can change the U value here to 6, 8 and 10
U = 8;

s = [0.1, 0.3, 0.5] # variance values used to create W_R

# You can change the phi value related to the U you set above. Ideally, phi should be smaller or equal to U
phi = [4, 6, 8]

lambda = 1 # we use two lambda before but decided to just report those of lambda  = 1 since there are no obvious difference.

iter_num = 100;
t_list = [0.05, 0.1, 0.15] # the t values used for the ILP problem - W_ILP = t * W + (1-t) * W_R


# ------------------------------------------------ Data Loading --------------------------------------------------
println("loading data...")

papers = readlines("../RevieweData/RelevenceJudgement/queryAspects.txt")
papers_topics = []
for i in 1:length(papers)
    if i % 2 == 0
        append!(papers_topics, [split(papers[i], " ")])
    end
end

papers_ = zeros(Int8, 73, 25)
for i in 1:73
    for j in 1:25
        if "T"*string(j) in papers_topics[i]
            papers_[i,j] = 1
        end
    end
end

reviewers = readlines("../RevieweData/RelevenceJudgement/Reviewers.txt")
reviewers_topics = []
for i in 1:length(reviewers)
    append!(reviewers_topics, [split(reviewers[i], " ")])
end
for i in reviewers_topics
    i[1] = split(i[1], "\t")[2]
end

reviewers_ = zeros(Int8, 189, 25)
for i in 1:189
    for j in 1:25
        if "T"*string(j) in reviewers_topics[i]
            reviewers_[i,j] = 1
        end
    end
end

# ------------------------------------------------ Get quality matrix W --------------------------------------------------

W = zeros(Float64, n, m) # Quality matrix after normalization
WW = zeros(Float64, n, m)
# f(papers_[1, :])
for i in 1:n
    for j in 1:m
        kp=f(papers_[i, :])
        kr=f(reviewers_[j, :])
        if kp*kr>0
            WW[i,j] = (papers_[i, :])' * (reviewers_[j, :])
            W[i,j] = WW[i,j]/(kr*kp)
        end
    end
end

println("solving standard ILP...")

model_II = Model(HiGHS.Optimizer)
set_optimizer_attribute(model_II, "output_flag", false)
@variable(model_II, X_ILP_standard[1:n, 1:m], Bin)
@objective(model_II, Max, sum(W .* X_ILP_standard))
@constraint(model_II, [i=1:n], sum(X_ILP_standard, dims = 2)[i] >= u1)
@constraint(model_II, [i=1:n], sum(X_ILP_standard, dims = 2)[i] <= u2)
@constraint(model_II, [i=1:m], sum(X_ILP_standard, dims = 1)[i] <= U);

optimize!(model_II)

X_ILP_standard=JuMP.value.(X_ILP_standard);

W_R = copy(W)
W_R = convert(Matrix{Float64}, W_R);

quality_ILP_1 = zeros(length(s), length(phi))
quality_ILP_2 = zeros(length(s), length(phi))
quality_ILP_3 = zeros(length(s), length(phi))
quality_BLP_ = zeros(length(s), length(phi))
fair_effort_ratio_1 = zeros(length(s), length(phi))
fair_effort_ratio_2 = zeros(length(s), length(phi))
fair_effort_ratio_3 = zeros(length(s), length(phi))
variance_ratio_1 = zeros(length(s), length(phi))
variance_ratio_2 = zeros(length(s), length(phi))
variance_ratio_3 = zeros(length(s), length(phi))

# ------------------------------------------------ Begin to solve --------------------------------------------------

# println("start running bilevel problem with the second lambda")
println("start ruinning loop...")

for xx in 1:length(s)
    for yy in 1:length(phi)

        sum_quality_ratio_ILP_1 = 0;
        sum_quality_ratio_ILP_2 = 0;
        sum_quality_ratio_ILP_3 = 0;

        sum_quality_ratio_BLP = 0

        sum_fair_effort_ratio_1 = 0;
        sum_fair_effort_ratio_2 = 0;
        sum_fair_effort_ratio_3 = 0;

        sum_var_ratio_1 = 0;
        sum_var_ratio_2 = 0;
        sum_var_ratio_3 = 0;

        # definition of the W_R as <Random> 
        for iter in 1:iter_num
            # ------------------------------------------------------ select the random distribution -----------------------------------------
            d = Uniform(0, 1)  # uniform distribution
            # d = Exponential(0.5) # exponential distribution

            for i in 1:n
                for j in 1:m
                    W_R[i,j] = abs(rand(d,1)[1])
                end
            end

            # ------------------------------------------------ solve ILP --------------------------------------------------
            println("solving ILP...") 

            max_quality = 0
            # X_ILP = 0
            # X_ILP = X_ILP_standard

            avr_effort_ILP_1 = 0;
            avr_effort_ILP_2 = 0;
            avr_effort_ILP_3 = 0;

            var_ILP_1 = 0;
            var_ILP_2 = 0;
            var_ILP_3 = 0;

            for t in t_list
                model = Model(HiGHS.Optimizer)
                W_ILP = (1-t) * W - t * W_R
                set_optimizer_attribute(model, "output_flag", false)
                @variable(model, X_ILP_t[1:n, 1:m], Bin)
                @objective(model, Max, sum(W_ILP .* X_ILP_t))
                @constraint(model, [i=1:n], sum(X_ILP_t, dims = 2)[i] >= u1)
                @constraint(model, [i=1:n], sum(X_ILP_t, dims = 2)[i] <= u2)
                @constraint(model, [i=1:m], sum(X_ILP_t, dims = 1)[i] <= U);
                optimize!(model)

                X_ILP_t=JuMP.value.(X_ILP_t);
                # println(X_ILP_t)

                if t == t_list[1]
                    total_effort_ILP_1 = sum(W_R .* X_ILP_t);
                    active_ILP_1 = sum(x->x!=0, sum(X_ILP_t, dims=1));
                    sum_quality_ratio_ILP_1 = sum_quality_ratio_ILP_1 + sum(W .* X_ILP_t)/sum(W .* X_ILP_standard);  # <<< for directly save
                    avr_effort_ILP_1 = total_effort_ILP_1/active_ILP_1;  # <<< for use in BLP to save
                    effort_vector_1 = sum(W_R .* X_ILP_t, dims = 1);
                    effort_vector_1 = deleteat!(vec(effort_vector_1), vec(effort_vector_1) .== 0);
                    var_ILP_1 = var(effort_vector_1)  # <<< for use in BLP to save
                elseif t == t_list[2]
                    total_effort_ILP_2 = sum(W_R .* X_ILP_t);
                    active_ILP_2 = sum(x->x!=0, sum(X_ILP_t, dims=1));
                    sum_quality_ratio_ILP_2 = sum_quality_ratio_ILP_2 + sum(W .* X_ILP_t)/sum(W .* X_ILP_standard);  # <<< for directly save
                    avr_effort_ILP_2 = total_effort_ILP_2/active_ILP_2;  # <<< for use in BLP to save
                    effort_vector_2 = sum(W_R .* X_ILP_t, dims = 1);
                    effort_vector_2 = deleteat!(vec(effort_vector_2), vec(effort_vector_2) .== 0);
                    var_ILP_2 = var(effort_vector_2)  # <<< for use in BLP to save
                else
                    total_effort_ILP_3 = sum(W_R .* X_ILP_t);
                    active_ILP_3 = sum(x->x!=0, sum(X_ILP_t, dims=1));
                    sum_quality_ratio_ILP_3 = sum_quality_ratio_ILP_3 + sum(W .* X_ILP_t)/sum(W .* X_ILP_standard);  # <<< for directly save
                    avr_effort_ILP_3 = total_effort_ILP_3/active_ILP_3;  # <<< for use in BLP to save
                    effort_vector_3 = sum(W_R .* X_ILP_t, dims = 1);
                    effort_vector_3 = deleteat!(vec(effort_vector_3), vec(effort_vector_3) .== 0);
                    var_ILP_3 = var(effort_vector_3)  # <<< for use in BLP to save
                end

            end

            # ------------------------------------------------ solve BLP --------------------------------------------------
            println("solving BLP...")

            #Build Z
            model_Z = Model(HiGHS.Optimizer)
            set_optimizer_attribute(model_Z, "output_flag", false)
            @variable(model_Z, Z[1:n, 1:m], Bin)
            @objective(model_Z, Max, sum(W .* Z))
            @constraint(model_Z, [i=1:m], sum(Z, dims = 1)[i] == U + phi[yy])
            optimize!(model_Z)
            Z=JuMP.value.(Z);

            model_1 = Model(HiGHS.Optimizer)
            set_optimizer_attribute(model_1, "output_flag", false)
            # model = Model(CPLEX.Optimizer)
            @variable(model_1, Y[1:n, 1:m], Bin)
            @objective(model_1, Min, sum(W_R .* Y))
            @constraint(model_1, [i=1:m], sum(Y, dims = 1)[i] >= min(U,sum(Z, dims = 1)[i]))
            for i in 1:n
                for j in 1:m
                    @constraint(model_1, Y[i,j] <= Z[i,j])
                end
            end

            optimize!(model_1)

            Y_ILP=JuMP.value.(Y);
            W_m=W+lambda*Y_ILP;

            model_2 = Model(HiGHS.Optimizer)
            set_optimizer_attribute(model_2, "output_flag", false)
            @variable(model_2, X[1:n, 1:m], Bin)
            @objective(model_2, Max, sum(W_m .* X))
            @constraint(model_2, [i=1:n], sum(X, dims = 2)[i] >= u1)
            @constraint(model_2, [i=1:n], sum(X, dims = 2)[i] <= u2)
            @constraint(model_2, [i=1:m], sum(X, dims = 1)[i] <= U);

            for i in 1:n
                for j in 1:m
                    @constraint(model_2, X[i,j] <=1 - Z[i,j] + Y_ILP[i,j]);
                end
            end

            optimize!(model_2);

            X_BLP = JuMP.value.(X);

            # ------------------------------------------------ save values for one iteration --------------------------------------------------

            ############ NEW VERSION

            total_effort = sum(W_R .* X_BLP);
            active_reviewers = sum(x->x!=0, sum(X_BLP, dims=1));
            sum_quality_ratio_BLP = sum_quality_ratio_BLP + sum(W .* X_BLP)/sum(W .* X_ILP_standard);  # <<< for directly save
            avr_effort = total_effort/active_reviewers;
            sum_fair_effort_ratio_1 = sum_fair_effort_ratio_1 + avr_effort/avr_effort_ILP_1;  # <<< for directly save
            sum_fair_effort_ratio_2 = sum_fair_effort_ratio_2 + avr_effort/avr_effort_ILP_2;  # <<< for directly save
            sum_fair_effort_ratio_3 = sum_fair_effort_ratio_3 + avr_effort/avr_effort_ILP_3;  # <<< for directly save
            effort_vector = sum(W_R .* X_BLP, dims = 1);
            effort_vector = deleteat!(vec(effort_vector), vec(effort_vector) .== 0);
            var_BLP = var(effort_vector);
            sum_var_ratio_1 = sum_var_ratio_1 + var_BLP/var_ILP_1;  # <<< for directly save
            sum_var_ratio_2 = sum_var_ratio_2 + var_BLP/var_ILP_2;  # <<< for directly save
            sum_var_ratio_3 = sum_var_ratio_3 + var_BLP/var_ILP_3;  # <<< for directly save

            println("number of iterations: "*string(iter))
        end

        # ------------------------------------------------ get the averaged final results --------------------------------------------------

        ########### NEW VERSION
        quality_ILP_1[xx,yy] = sum_quality_ratio_ILP_1 / iter_num;
        quality_ILP_2[xx,yy] = sum_quality_ratio_ILP_2 / iter_num;
        quality_ILP_3[xx,yy] = sum_quality_ratio_ILP_3 / iter_num;
        quality_BLP_[xx,yy] = sum_quality_ratio_BLP / iter_num;
        fair_effort_ratio_1[xx, yy] = sum_fair_effort_ratio_1 / iter_num;
        fair_effort_ratio_2[xx, yy] = sum_fair_effort_ratio_2 / iter_num;
        fair_effort_ratio_3[xx, yy] = sum_fair_effort_ratio_3 / iter_num;
        variance_ratio_1[xx, yy] = sum_var_ratio_1 / iter_num;
        variance_ratio_2[xx, yy] = sum_var_ratio_2 / iter_num;
        variance_ratio_3[xx, yy] = sum_var_ratio_3 / iter_num;
        println("number of s-phi settings: "*string(xx)*"-"*string(yy))

        # ------------------------------------------------ write into file --------------------------------------------------
        open("results_file_random_U_8.txt", "w") do io
            write(io, "lambda = 1, U = 8, W_R is the uniform random case")
            write(io, "\n")
            write(io, "    phi_1 - phi_2 - phi3")
            write(io, "\n")
            write(io, "s1")
            write(io, "\n")
            write(io, "s2")
            write(io, "\n")
            write(io, "s3")
            write(io, "\n")
            write(io, "quality ratio BLP / standard ILP")
            write(io, "\n")
            writedlm(io, quality_BLP_, ',')
            write(io, "\n")

            write(io, "quality ratio ILP - t1 / standard ILP")
            write(io, "\n")
            writedlm(io, quality_ILP_1, ',')
            write(io, "\n")

            write(io, "quality ratio ILP - t2 / standard ILP")
            write(io, "\n")
            writedlm(io, quality_ILP_2, ',')
            write(io, "\n")

            write(io, "quality ratio ILP - t3 / standard ILP")
            write(io, "\n")
            writedlm(io, quality_ILP_3, ',')
            write(io, "\n")

            write(io, "fair effort ratio - BLP / ILP_t1")
            write(io, "\n")
            writedlm(io, fair_effort_ratio_1, ',')
            write(io, "\n")

            write(io, "fair effort ratio - BLP / ILP_t2")
            write(io, "\n")
            writedlm(io, fair_effort_ratio_2, ',')
            write(io, "\n")

            write(io, "fair effort ratio - BLP / ILP_t3")
            write(io, "\n")
            writedlm(io, fair_effort_ratio_3, ',')
            write(io, "\n")

            write(io, "vaiance ratio - BLP / ILP_t1")
            write(io, "\n")
            writedlm(io, variance_ratio_1, ',')
            write(io, "\n")

            write(io, "vaiance ratio - BLP / ILP_t2")
            write(io, "\n")
            writedlm(io, variance_ratio_2, ',')
            write(io, "\n")

            write(io, "vaiance ratio - BLP / ILP_t3")
            write(io, "\n")
            writedlm(io, variance_ratio_3, ',')
            write(io, "\n")
        end;
    end
end

println("completed with lambda = 1")
println("all completed.")