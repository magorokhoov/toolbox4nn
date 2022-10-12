class AccumulationStats:
    def __init__(self) -> None:
        self.count = 0.000001 # to prevent division by zero
        self.accumulation = {}

        self.reset_accumulation()

    def reset_accumulation(self) -> None:
        for accumulation_name in self.accumulation:
            self.accumulation[accumulation_name] = 0.0

        self.count = 0.000001 # to prevent division by zero

    def add_accumulation(self, add: dict) -> None:
        self.count += 1
        for add_name in add:
            if add_name not in self.accumulation:
                self.accumulation[add_name] = 0.0 # lazy initialization
            
            self.accumulation[add_name] += add[add_name]

    def get_accumulation(self) -> dict:
        return self.accumulation.copy()

    def get_str(self, reset=True) -> str:
        result_str = ''
        for i, acc_name in enumerate(sorted(self.accumulation)):
            mean_acc = self.accumulation[acc_name] / self.count
            result_str += f'{acc_name:s}={mean_acc:.3e}'
            if i < len(self.accumulation) - 1:
                result_str += ', '

        if reset:
            self.reset_accumulation()

        return result_str