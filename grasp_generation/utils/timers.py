from __future__ import annotations
import time
from tqdm import tqdm
from collections import defaultdict
from typing import DefaultDict, List, Optional
import pandas as pd


class SectionTimer:
    def __init__(self, section_name: str, print_timing: bool = False):
        self.section_name = section_name
        self.print_timing = print_timing

        self.start_time = None
        self.end_time = None

    def start(self) -> SectionTimer:
        self.start_time = time.perf_counter_ns()
        return self

    def stop(self) -> SectionTimer:
        self.end_time = time.perf_counter_ns()
        return self

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        if self.print_timing:
            print(
                f"Time elapsed for '{self.section_name}' is {self.elapsed_time_ms} ms"
            )
        return

    @property
    def elapsed_time_ns(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started yet.")
        if self.end_time is None:
            raise ValueError("Timer has not been stopped yet.")
        return self.end_time - self.start_time

    @property
    def elapsed_time_ms(self):
        return self.elapsed_time_ns / 1e6

    @property
    def elapsed_time_s(self):
        return self.elapsed_time_ns / 1e9


class LoopTimer:
    def __init__(self):
        self.reset_section_timers()

    def reset_section_timers(self):
        self.section_names_to_timers: DefaultDict[
            str, List[SectionTimer]
        ] = defaultdict(list)

    def add_section_timer(self, section_name: str) -> SectionTimer:
        new_section_timer = SectionTimer(section_name)
        self.section_names_to_timers[section_name].append(new_section_timer)
        return new_section_timer

    def get_section_times_df(self) -> pd.DataFrame:
        section_name_to_most_recent_time_ms = {
            section_name: section_timers[-1].elapsed_time_ms
            for section_name, section_timers in self.section_names_to_timers.items()
        }
        most_recent_total_time_ms = sum(section_name_to_most_recent_time_ms.values())

        section_name_to_sum_time_ms = {
            section_name: sum(
                section_timer.elapsed_time_ms for section_timer in section_timers
            )
            for section_name, section_timers in self.section_names_to_timers.items()
        }
        total_time_ms = sum(section_name_to_sum_time_ms.values())
        df = pd.DataFrame(
            {
                "Section": list(section_name_to_most_recent_time_ms.keys()),
                "Most Recent Time (ms)": list(
                    section_name_to_most_recent_time_ms.values()
                ),
                "Most Recent Time (%)": [
                    t / most_recent_total_time_ms * 100
                    for t in section_name_to_most_recent_time_ms.values()
                ],
                "Sum Time (ms)": list(section_name_to_sum_time_ms.values()),
                "Sum Time (%)": [
                    t / total_time_ms * 100
                    for t in section_name_to_sum_time_ms.values()
                ],
            }
        )
        return df

    def pretty_print_section_times(
        self, df: Optional[pd.DataFrame] = None, n_decimal_places: int = 1
    ) -> None:
        if df is None:
            df = self.get_section_times_df()
        print(df.to_markdown(floatfmt=f".{n_decimal_places}f"))


def main() -> None:
    loop_timer = LoopTimer()
    for i in (pbar := tqdm(range(100))):
        with loop_timer.add_section_timer("test"):
            if i < 3:
                time.sleep(1.0)
            else:
                time.sleep(0.1)

        with loop_timer.add_section_timer("test2"):
            time.sleep(0.3)

        section_times_df = loop_timer.get_section_times_df()
        loop_timer.pretty_print_section_times(df=section_times_df)
        pbar.set_description(
            " | ".join(
                [
                    f"{section_times_df['Section'].iloc[j]}: {section_times_df['Most Recent Time (ms)'].iloc[j]:.0f}"
                    for j in range(len(section_times_df))
                ]
            )
        )


if __name__ == "__main__":
    main()
