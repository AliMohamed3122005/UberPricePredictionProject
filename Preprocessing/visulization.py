import matplotlib.pyplot as plt


def plot_fare_distribution(df):
    plt.style.use("dark_background")

    plt.figure(figsize=(8, 5))

    df["fare_amount"].hist(
        color="cyan",
        bins=50,
        edgecolor="black"
    )

    plt.title("Fare Amount Distribution", fontsize=13, color="cyan")
    plt.xlabel("Fare Amount", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(color="gray", alpha=0.25)

    plt.show()


def plot_distance_distribution(df):
    plt.style.use("dark_background")

    plt.figure(figsize=(8, 5))

    df["dist_travel_km"].hist(
        color="lime",
        bins=50,
        edgecolor="black",
        range=(0, 25)
    )

    plt.title("Distance Distribution", fontsize=13, color="lime")
    plt.xlabel("Distance KM", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.xlim(0, 25)
    plt.grid(color="gray", alpha=0.25)

    plt.show()


def plot_fare_vs_distance(df):
    plt.style.use("dark_background")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["dist_travel_km"],
        df["fare_amount"],
        color="magenta",
        alpha=0.3
    )

    plt.title("Fare Amount vs Distance", fontsize=13, color="magenta")
    plt.xlabel("Distance KM", fontsize=11)
    plt.ylabel("Fare Amount", fontsize=11)
    plt.grid(color="gray", alpha=0.25)

    plt.show()


def plot_trips_by_hour(df):
    plt.style.use("dark_background")

    trips_by_hour = df["hour"].value_counts().sort_index()

    plt.figure(figsize=(9, 5))

    plt.bar(
        trips_by_hour.index,
        trips_by_hour.values,
        color="cyan",
        edgecolor="black"
    )

    plt.title("Trips by Hour", fontsize=13, color="cyan")
    plt.xlabel("Hour", fontsize=11)
    plt.ylabel("Number of Trips", fontsize=11)
    plt.xticks(range(0, 24))
    plt.grid(axis="y", color="gray", alpha=0.25)

    plt.show()


def plot_avg_fare_by_hour(df):
    plt.style.use("dark_background")

    avg_fare_by_hour = df.groupby("hour")["fare_amount"].mean()

    plt.figure(figsize=(9, 5))

    plt.plot(
        avg_fare_by_hour.index,
        avg_fare_by_hour.values,
        color="lime",
        marker="o",
        linewidth=2
    )

    plt.title("Average Fare by Hour", fontsize=13, color="lime")
    plt.xlabel("Hour", fontsize=11)
    plt.ylabel("Average Fare", fontsize=11)
    plt.xticks(range(0, 24))
    plt.grid(color="gray", alpha=0.25)

    plt.show()


def plot_passenger_count(df):
    plt.style.use("dark_background")

    passenger_counts = df["passenger_count"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))

    plt.bar(
        passenger_counts.index,
        passenger_counts.values,
        color="cyan",
        edgecolor="black"
    )

    plt.title("Trips by Passenger Count", fontsize=13, color="cyan")
    plt.xlabel("Passenger Count", fontsize=11)
    plt.ylabel("Number of Trips", fontsize=11)
    plt.grid(axis="y", color="gray", alpha=0.25)

    plt.show()


def plot_all_preprocessing_visuals(df):
    plot_fare_distribution(df)
    plot_passenger_count(df)

    if "dist_travel_km" in df.columns:
        plot_distance_distribution(df)
        plot_fare_vs_distance(df)

    if "hour" in df.columns:
        plot_trips_by_hour(df)
        plot_avg_fare_by_hour(df)