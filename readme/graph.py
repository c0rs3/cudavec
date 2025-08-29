import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

def read_and_plot(filename):
    sizes = []
    cublas_times = []
    cuda_times = []
    flat_times = []
    avx_times = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    assert len(lines) % 5 == 0, "Log file must be in blocks of 3 lines."

    for i in range(0, len(lines), 5):
        try:
            size = int(lines[i].strip())
            cublas_time = float(lines[i + 1].strip())
            cuda_time = float(lines[i + 2].strip())
            flat_time = float(lines[i + 3].strip())
            avx_time = float(lines[i + 4].strip())
        except ValueError:
            print(f"Skipping malformed entry at lines {i}-{i + 2}")
            continue

        sizes.append(size)
        cublas_times.append(cublas_time)
        cuda_times.append(cuda_time)
        flat_times.append(flat_time)
        avx_times.append(avx_time)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cublas_times, label='cuBLAS Runtime (ms)', marker='o')
    plt.plot(sizes, cuda_times, label='CUDA Kernel Runtime (ms)', marker='o')
    plt.plot(sizes, flat_times, label='Flat (CPU) Runtime (ms)', marker='o')
    plt.plot(sizes, avx_times, label='AVX Runtime (ms)', marker='o')

    plt.xlabel('Matrix Size (Total Elements)')
    plt.ylabel('Runtime (miliseconds)')
    plt.title('Runtime Comparsion')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Log scales for both axes
    plt.xscale('log')
    plt.yscale('log')

    ax = plt.gca()

    # Increase Y-axis tick density on log scale
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())  # Hide cluttered minor tick labels

    plt.tight_layout()
    plt.show()

# Usage
if __name__ == "__main__":
    # name = input("Enter file name")
    
    read_and_plot("log.txt")
