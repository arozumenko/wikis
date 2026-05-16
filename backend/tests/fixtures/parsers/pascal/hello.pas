program HelloDemo;

type
    TGreeter = class
        procedure Greet(name: string);
        procedure FormatName(n: string);
    end;

procedure TGreeter.Greet(name: string);
begin
    FormatName(name);
end;

procedure TGreeter.FormatName(n: string);
begin
    WriteLn(n);
end;

begin
    WriteLn('start');
end.
