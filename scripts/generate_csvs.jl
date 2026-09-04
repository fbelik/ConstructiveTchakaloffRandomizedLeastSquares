using TchakaloffResults

function generate_csvs(output_directory="data/")
    trial_run(; save_to=output_directory)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_csvs(ARGS...)
end